# As seen on https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding:
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # positional encoding
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        x = self.dropout(x)


class TransformerLM(nn.Module):
    def __init__(self, n_tokens, n_inp, n_head, n_hid, n_layers, dropout=0.5):
        super(TransformerLM, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(n_inp, dropout)
        encoder_layers = TransformerEncoderLayer(n_inp, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(n_inp, n_head, n_hid, dropout)
        self.encoder = nn.Embedding(n_tokens, n_inp)
        self.n_inp = n_inp
        self.decoder = nn.Linear(n_inp, n_tokens)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def _generate_square_subsequent_mask(self, sz):
        # torch.triu returns the upper triangular part of a matrix
        mask = (torch.tri(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        # Why mulyiply? Maybe to cancel out the scaling factor?
        src = self.encoder(src) * math.sqrt(self.n_inp)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, self.src_mask)
        out = self.decoder(out)
        return out
