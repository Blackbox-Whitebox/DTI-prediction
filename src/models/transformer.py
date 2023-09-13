"""
All Models that make use of Transformers here.
"""

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding
    """
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        positions = torch.arange(0, max_len).unsqueeze(1)

        frequencies = 10000**(torch.arange(0, d_model, 2)/d_model)
        
        self.encoding = torch.zeros(max_len, d_model)

        self.encoding[:, 1::2] = torch.sin(positions / frequencies)
        self.encoding[:, 0::2] = torch.cos(positions / frequencies)

    def forward(self, x):
        seq_len = x.shape[1]

        x = x  + self.encoding[:seq_len].to(x.device)

        return x


class CrossLayer(nn.Module):
    def __init__(self, embed_size, n_head, hidden_size, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.self_attention_encoder_1 = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )

        self.self_attention_encoder_2 = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attention1 = nn.MultiheadAttention(embed_dim=embed_size, num_heads=n_head, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(embed_dim=embed_size, num_heads=n_head, batch_first=True)


    def forward(self, x1, x2):

        # x1 -> (N, S)
        # x2 -> (N, S)

        cross_alignment_1, _ = self.cross_attention1(
            query=x2,
            key=x1,
            value=x1
        )

        cross_alignment_2, _ = self.cross_attention2(
            query=x1,
            key=x2,
            value=x2
        )

        alignment_1 = self.self_attention_encoder_1(cross_alignment_1)
        alignment_2 = self.self_attention_encoder_2(cross_alignment_2)

        return alignment_1, alignment_2


class CrossNet(nn.Module):
    def __init__(self, hidden_size, n_head, num_layers, dropout, mode="classification"):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_head = n_head
        self.num_layers = num_layers
        self.dropout = dropout
        self.is_contrastive = mode=="contrastive"

        self.drug_embedding = None
        self.protein_embedding = None

        self.cross_layers = nn.ModuleList([
            CrossLayer(embed_size=64, n_head=n_head, hidden_size=hidden_size, dropout=dropout) for _ in range(num_layers)
        ])

        if not self.is_contrastive:
            self.out = nn.Linear(64*2, 2)


        self.positional_encoding = PositionalEncoding(d_model=64, max_len=20000)


    def set_drug_protein_embeddings(self, drug, protein):
        self.drug_embedding = drug
        self.protein_embedding = protein

        self.drug_embedding.requires_grad = False
        self.protein_embedding.requires_grad = False


    def forward(self, x):
        x1, x2 = x
        N = x1.shape[0]

        x1 = self.drug_embedding(x1) # (N, S, E)
        x2 = self.protein_embedding(x2) # (N, S, E)

        x1 = self.positional_encoding(x1)
        x2 = self.positional_encoding(x2)

        for layer in self.cross_layers:
            x1, x2 = layer(x1, x2)


        # x1 (N, S, 64)
        # x2 (N, S, 64)

        x1 = x1.mean(1) # (N, 64)
        x2 = x2.mean(1) # (N, 64)

        if self.is_contrastive: 
            return x1, x2
        
        else:
            x = torch.cat([x1, x2], 1) # (N, 64*2)
            x = self.out(x)

            return x


class Transformer(nn.Module):
    def __init__(self, hidden_size, n_head, num_layers, dropout, mode=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.drug_embedding = None
        self.protein_embedding = None

        self.transformer = nn.Transformer(
            d_model=64,
            nhead=n_head,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )

        self.positional_encoding = PositionalEncoding(d_model=64, max_len=20000)

        self.out = nn.Linear(64, 2)


    def set_drug_protein_embeddings(self, drug, protein):
        self.drug_embedding = drug
        self.protein_embedding = protein

        self.drug_embedding.requires_grad = False
        self.protein_embedding.requires_grad = False


    def forward(self, x):
        x1, x2 = x
        
        N = x1.shape[0]

        x1 = self.drug_embedding(x1) # (N, S, E)
        x2 = self.protein_embedding(x2) # (N, S, E)

        x1 = self.positional_encoding(x1)
        x2 = self.positional_encoding(x2)

        x = self.transformer(x1, x2)

        x = x.mean(1)

        out = self.out(x)
        
        return out


def build(name, config):
    hidden_size = config["HIDDEN_SIZE"]
    num_layers = config["NUM_LAYERS"]
    dropout = config["DROPOUT"]
    num_heads = config["NUM_HEADS"]
    mode = "contrastive" if bool(config["CONTRASTIVE"]) else "classification"

    
    if "cross" in name:
        net = CrossNet(hidden_size=hidden_size, n_head=num_heads, num_layers=num_layers, dropout=dropout, mode=mode)

    else:
        net = Transformer(hidden_size=hidden_size, n_head=num_heads, num_layers=num_layers, dropout=dropout)

    return net