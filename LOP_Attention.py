import torch
import torch.nn as nn
import torch.nn.functional as F

class LOP_Adaptive(nn.Module):
    """
    Multi-head attention + head-wise Conv1d với:
     - kernel_size động theo importance của head
     - smart padding (edge replication) để giảm edge artifacts
    Ý tưởng:
     - Tạo conv kernel với kích thước tối đa max_k.
     - Với mỗi head lấy một slice trung tâm của conv.weight để mô phỏng kernel size k <= max_k.
     - Kernel động k được suy ra từ attention importance (trên batch).
    """
    def __init__(self, input_dim, output_dim, num_heads, min_kernel, max_kernel):
        super().__init__()
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.last_attn = None  # thêm dòng này


        # Projection
        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)

    
        self.max_kernel = max_kernel if max_kernel % 2 == 1 else max_kernel + 1  
        self.min_kernel = min_kernel if min_kernel % 2 == 1 else min_kernel + 1
        assert 1 <= self.min_kernel <= self.max_kernel, "min_kernel must be >=1 and <= max_kernel"

        self.convs = nn.ModuleList([
            nn.Conv1d(self.head_dim, self.head_dim, kernel_size=self.max_kernel, padding=0, bias=True)
            for _ in range(num_heads)
        ])


        self.kernel_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.fc = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def _smart_pad(self, x, target_len):
        """
        Pad tensor x (B, C, L) to length target_len using edge replication.
        If L >= target_len -> do nothing.
        """
        B, C, L = x.shape
        if L >= target_len:
            return x
        needed = target_len - L
        left = needed // 2
        right = needed - left
        # replicate edges
        left_pad = x[:, :, 0:1].repeat(1, 1, left) if left > 0 else torch.empty(B, C, 0, device=x.device, dtype=x.dtype)
        right_pad = x[:, :, -1:].repeat(1, 1, right) if right > 0 else torch.empty(B, C, 0, device=x.device, dtype=x.dtype)
        return torch.cat([left_pad, x, right_pad], dim=2)

    def _central_slice(self, weight, k):
        _, _, max_k = weight.shape
        assert max_k >= k
        center = max_k // 2
        half = k // 2
        start = center - half
        end = center + half + 1
        return weight[:, :, start:end]

    def forward(self, x,return_attn=False):
        """
        x: (B, T, input_dim)
        returns: (B, T, output_dim)
        """
        B, T, _ = x.size()
        # Linear projections and reshape into heads
        q = self.q(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, d)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = F.softmax(attn_logits, dim=-1)


        self.last_attn = attn.detach().cpu()
        importance = attn.mean(dim=(-2, -1))  # (B, H)
        importance = importance.mean(dim=0, keepdim=True).transpose(0, 1)  # (H, 1)
        # normalize to [0,1]
        imp_min = importance.min()
        imp_max = importance.max()
        imp_norm = (importance - imp_min) / (imp_max - imp_min + 1e-6)  # (H,1)

        imp_scaled = self.kernel_proj(imp_norm)  # (H,1)
        frac = torch.sigmoid(imp_scaled).squeeze(-1)  # (H,)
        kernel_range = (self.max_kernel - self.min_kernel)
        k_values = (self.min_kernel + (frac * kernel_range)).round().to(torch.int64)  # (H,)
        k_values = k_values + (1 - (k_values % 2))
        k_values = torch.clamp(k_values, min=self.min_kernel, max=self.max_kernel)

        out = torch.matmul(attn, v)  # (B, H, T, d)

        conv_outs = []
        for i, conv in enumerate(self.convs):
            k_i = int(k_values[i].item())  # desired kernel for head i (odd)
            head = out[:, i].transpose(1, 2)  # (B, d, T)
            head_padded = self._smart_pad(head, max(T, k_i))
            w = conv.weight  # (out_ch=head_dim, in_ch=head_dim, max_k)
            b = conv.bias
            w_sliced = self._central_slice(w, k_i)  # (d, d, k_i)
            head_conv = F.conv1d(head_padded, w_sliced, bias=b, padding=0, groups=1)

            L_out = head_conv.shape[-1]
            if L_out > T:
                start = (L_out - T) // 2
                head_conv = head_conv[:, :, start:start + T]
            elif L_out < T:
                head_conv = self._smart_pad(head_conv, T)
            head_out = head_conv.transpose(1, 2)
            conv_outs.append(head_out)

        out_cat = torch.cat(conv_outs, dim=-1)
        out_proj = self.fc(out_cat)
        out_proj = self.norm(out_proj + x)
        if return_attn:
            return out_proj, attn  # shape B x H x T x T
        return out_proj

class LOP_Attention(nn.Module):
    """
    Transformer Encoder block sử dụng MultiHeadConvAttentionAdaptive + FeedForward.
    Có residual connection và layer normalization.
    """
    def __init__(self, input_dim, num_heads, ff_dim, dropout,
                 min_kernel, max_kernel):
        super().__init__()

        # Dùng attention có kernel động
        self.attn = LOP_Adaptive(
            input_dim=input_dim,
            output_dim=input_dim,
            num_heads=num_heads,
            min_kernel=min_kernel,
            max_kernel=max_kernel
        )

        # Feedforward block chuẩn Transformer
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim)
        )

        # LayerNorm + dropout
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        attn_out = self.attn(x, return_attn=return_attn)
        if return_attn:
            out_attn = attn_out[1]
            attn_out = attn_out[0]
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        if return_attn:
            return x, out_attn
        return x
