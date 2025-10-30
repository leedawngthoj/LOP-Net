import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLSTMCell_r(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Linear cho input và hidden
        self.W_x = nn.Linear(input_size, 5 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 5 * hidden_size)
        self.W_y = nn.Linear(hidden_size, hidden_size)
        
        # LayerNorm riêng cho r_t và c_prev
        self.norm_r = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        
        self.drop = nn.Dropout(dropout)

    def forward(self, x_t, h_prev, c_prev):
        # Gates
        gates = self.W_x(x_t) + self.W_h(h_prev)
        i_t, f_t, g_t, o_t, r_t_raw = torch.split(gates, self.hidden_size, dim=1)
        
        # Standard LSTM activations
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = self.drop(torch.tanh(g_t))
        o_t = torch.sigmoid(o_t)

        # chuẩn hoá c_prev trước khi tính r_t
        y_prev = torch.tanh(self.norm_c(c_prev))
        r_t = torch.sigmoid(self.norm_r(r_t_raw + self.W_y(y_prev)))

        # Update cell state with refined formula
        c_keep = (f_t + r_t * (1 - f_t)) * c_prev
        c_add  = (1 - r_t) * (i_t * g_t)
        c_t = c_keep + c_add

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class CustomLSTM_r(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = CustomLSTMCell_r(input_size, hidden_size)

    def forward(self, x, h0=None, c0=None):
        batch_size, seq_len, _ = x.size()
        h_size = self.cell.hidden_size
        if h0 is None:
            h_prev = torch.zeros(batch_size, h_size, device=x.device)
        else:
            h_prev = h0
        if c0 is None:
            c_prev = torch.zeros(batch_size, h_size, device=x.device)
        else:
            c_prev = c0

        outputs = []
        for t in range(seq_len):
            h_prev, c_prev = self.cell(x[:, t, :], h_prev, c_prev)
            outputs.append(h_prev.unsqueeze(1))
        return torch.cat(outputs, dim=1), (h_prev, c_prev)


class Relationship_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forward_lstm = CustomLSTM_r(input_size, hidden_size)
        self.backward_lstm = CustomLSTM_r(input_size, hidden_size)

    def forward(self, x):
        out_f, (h_f, c_f) = self.forward_lstm(x)
        out_b, (h_b, c_b) = self.backward_lstm(torch.flip(x, dims=[1]))
        out_b = torch.flip(out_b, dims=[1])
        out = torch.cat([out_f, out_b], dim=-1)
        h = torch.cat([h_f, h_b], dim=-1)
        c = torch.cat([c_f, c_b], dim=-1)
        return out, (h, c)
