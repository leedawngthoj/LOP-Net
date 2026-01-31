import torch
import torch.nn as nn
import torch.nn.functional as F
from LOP_Attention import LOP_Attention
from Relationship_LSTM import Relationship_LSTM
class LOP_Net(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=128,
        num_heads=4,
        ff_dim=64,
        rnn_hidden=128,
        dropout=0.01,
        fixed_radius=None 
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)

        # LOP_Attention block
        self.lop_attention = LOP_Attention(
            dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            fixed_radius=fixed_radius
        )

        # Relationship LSTM block
        self.relationship_lstm = Relationship_LSTM(embed_dim, rnn_hidden)

        self.norm = nn.LayerNorm(rnn_hidden * 2)
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, return_internal=False):
        x = self.embedding(x)
        x, _ = self.lop_attention(x, return_attn=True)

        out, (h, _), r_f, r_b, ck_f, ca_f = self.relationship_lstm(x)
        h = self.norm(h)
        logits = self.fc(h)

        if return_internal:
            return logits, {
                "r_f": r_f,
                "r_b": r_b,
                "c_keep": ck_f,
                "c_add": ca_f
            }

        return logits
