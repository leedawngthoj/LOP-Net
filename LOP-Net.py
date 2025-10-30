import torch
import torch.nn as nn
import torch.nn.functional as F
from LOP_Attention import LOP_Attention
from Relationship_LSTM import Relationship_LSTM
class LOP_Net(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, ff_dim=64, rnn_hidden=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)

        self.transformer = LOP_Attention(embed_dim, num_heads, ff_dim, dropout =0.3, min_kernel =3, max_kernel=11)

        # Bi-directional Relationship LSTM
        self.rnn = Relationship_LSTM(embed_dim, rnn_hidden)

        # LayerNorm cuối trước khi FC
        self.norm_final = nn.LayerNorm(rnn_hidden * 2)

        # Fully connected output
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # [B, T, embed_dim]

        # MultiHeadConvAttention block
        x, attn = self.transformer(x, return_attn=True)

        # Bi-directional Relationship LSTM
        _, (h, _) = self.rnn(x)
        h = self.norm_final(h)

        # FC output
        out = self.fc(h)
        return out


