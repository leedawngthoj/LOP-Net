import torch
import torch.nn as nn
import torch.nn.functional as F
class CustomLSTMCell_r(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_x = nn.Linear(input_size, 5 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 5 * hidden_size)
        self.W_y = nn.Linear(hidden_size, hidden_size)

        self.norm_r = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)

        self.drop = nn.Dropout(dropout)

        # for analysis
        self.last_r_t = None
        self.last_c_keep = None
        self.last_c_add = None

    def forward(self, x_t, h_prev, c_prev):
        gates = self.W_x(x_t) + self.W_h(h_prev)
        i_t, f_t, g_t, o_t, r_raw = torch.split(
            gates, self.hidden_size, dim=1
        )

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = self.drop(torch.tanh(g_t))
        o_t = torch.sigmoid(o_t)

        y_prev = torch.tanh(self.norm_c(c_prev))
        r_t = torch.sigmoid(self.norm_r(r_raw + self.W_y(y_prev)))

        # memory update
        c_keep = (f_t + r_t * (1 - f_t)) * c_prev
        c_add  = (1 - r_t) * (i_t * g_t)
        c_t = c_keep + c_add

        h_t = o_t * torch.tanh(c_t)

        # ===== logging =====
        self.last_r_t = r_t.detach().cpu()
        self.last_c_keep = c_keep.detach().cpu()
        self.last_c_add  = c_add.detach().cpu()

        return h_t, c_t
class CustomLSTM_r(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = CustomLSTMCell_r(input_size, hidden_size)

    def forward(self, x, h0=None, c0=None):
        B, T, _ = x.size()
        H = self.cell.hidden_size

        h = torch.zeros(B, H, device=x.device) if h0 is None else h0
        c = torch.zeros(B, H, device=x.device) if c0 is None else c0

        outputs, r_seq, c_keep_seq, c_add_seq = [], [], [], []

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h.unsqueeze(1))
            r_seq.append(self.cell.last_r_t.unsqueeze(1))
            c_keep_seq.append(self.cell.last_c_keep.unsqueeze(1))
            c_add_seq.append(self.cell.last_c_add.unsqueeze(1))

        return (
            torch.cat(outputs, dim=1),
            (h, c),
            torch.cat(r_seq, dim=1),
            torch.cat(c_keep_seq, dim=1),
            torch.cat(c_add_seq, dim=1),
        )
class Relationship_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forward_lstm = CustomLSTM_r(input_size, hidden_size)
        self.backward_lstm = CustomLSTM_r(input_size, hidden_size)

    def forward(self, x):
        out_f, (h_f, c_f), r_f, ck_f, ca_f = self.forward_lstm(x)

        out_b, (h_b, c_b), r_b, ck_b, ca_b = self.backward_lstm(
            torch.flip(x, dims=[1])
        )
        out_b = torch.flip(out_b, dims=[1])
        r_b   = torch.flip(r_b, dims=[1])
        ck_b  = torch.flip(ck_b, dims=[1])
        ca_b  = torch.flip(ca_b, dims=[1])

        out = torch.cat([out_f, out_b], dim=-1)
        h = torch.cat([h_f, h_b], dim=-1)
        c = torch.cat([c_f, c_b], dim=-1)

        return out, (h, c), r_f, r_b, ck_f, ca_f
