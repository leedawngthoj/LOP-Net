import torch
import torch.nn as nn
import torch.nn.functional as F

class LOP_Adaptive(nn.Module):
    """
    Multi-head self-attention + head-wise Conv1D
    with fully learnable, continuous receptive field per head.
    (Gaussian soft mask – stable gradient)
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        min_kernel=3,
        max_kernel=11,
        sigma=1.5,
        fixed_radius=None
    ):
        super().__init__()
        assert output_dim % num_heads == 0
        assert min_kernel % 2 == 1 and max_kernel % 2 == 1

        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.max_radius = max_kernel // 2
        self.sigma = sigma

        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)

        self.radius_param = nn.Parameter(
            torch.linspace(
                self.min_kernel // 2,
                self.max_kernel // 2,
                steps=num_heads
            )
        )

        self.radius_scale = nn.Parameter(torch.ones(num_heads))
        self.fixed_radius = fixed_radius
        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.head_dim,
                self.head_dim,
                kernel_size=self.max_kernel,
                padding=0,
                bias=True
            )
            for _ in range(num_heads)
        ])

        self.fc = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

        self.last_k_values = None
        self.last_radius = None
        self.last_attn = None

    def _smart_pad(self, x, target_len):
        B, C, L = x.shape
        if L >= target_len:
            return x
        pad = target_len - L
        left = pad // 2
        right = pad - left
        left_pad = x[:, :, :1].repeat(1, 1, left)
        right_pad = x[:, :, -1:].repeat(1, 1, right)
        return torch.cat([left_pad, x, right_pad], dim=-1)
   
    def radius_diversity_loss(radius):
        diff = radius.unsqueeze(0) - radius.unsqueeze(1)
        return -torch.mean(diff ** 2)


    def forward(self, x, return_attn=False):
        
        q = self.q(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn_logits, dim=-1)
        self.last_attn = attn.detach().cpu()

        out = torch.matmul(attn, v)  # (B, H, T, d)

        center = self.max_kernel // 2
        pos = torch.arange(self.max_kernel, device=x.device).float()
        pos = (pos - center).abs()  # (K,)

        conv_outs = []
        k_values = []
        radius_values = []

        for h in range(self.num_heads):
            if self.fixed_radius is None:
                r = self.radius_param[h] * self.radius_scale[h]
            else:
                r = torch.tensor(
                    self.fixed_radius,
                    device=x.device,
                    dtype=torch.float32
                )

            r = torch.clamp(r, 0.0, float(self.max_radius))
            
            mask = torch.exp(- (pos - r) ** 2 / (2 * self.sigma ** 2))
            mask = mask.view(1, 1, -1)

            w = self.convs[h].weight * mask
            b = self.convs[h].bias

            head = out[:, h].transpose(1, 2)  # (B, d, T)
            head = self._smart_pad(head, T + self.max_kernel)

            y = F.conv1d(head, w, b)

            start = (y.size(-1) - T) // 2
            y = y[:, :, start:start + T]

            conv_outs.append(y.transpose(1, 2))

            # logging only
            k_eff = 2 * torch.round(r).clamp(0, self.max_radius) + 1
            k_values.append(k_eff)
            radius_values.append(r)

        self.last_k_values = torch.stack(k_values).detach().cpu()
        self.last_radius = torch.stack(radius_values).detach().cpu()

        out = torch.cat(conv_outs, dim=-1)
        out = self.fc(out)
        out = self.norm(out + x)

        if return_attn:
            return out, attn
        return out
class LOP_Attention(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.01,fixed_radius=None):
        super().__init__()

        self.attn = LOP_Adaptive(
            dim, dim, num_heads,
            fixed_radius=fixed_radius
        )

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        if return_attn:
            y, attn = self.attn(x, return_attn=True)
        else:
            y = self.attn(x)

        x = self.norm1(x + self.dropout(y))
        y = self.ff(x)
        x = self.norm2(x + self.dropout(y))

        if return_attn:
            return x, attn
        return x
