import torch
import torch.nn as nn
import torch.nn.functional as F

class HTBILayer(nn.Module):

    def __init__(self, hidden_size, num_heads, ff_dim, dropout):
        super().__init__()
        self.bottleneck_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.ca_emo = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ca_int = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.bn_ln1 = nn.LayerNorm(hidden_size)
        self.bn_ln2 = nn.LayerNorm(hidden_size)
        self.bn_ffn = nn.Sequential(nn.Linear(hidden_size, ff_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_dim, hidden_size))
        self.ca_back_emo = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ca_back_int = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.emo_ln1 = nn.LayerNorm(hidden_size)
        self.emo_ln2 = nn.LayerNorm(hidden_size)
        self.emo_ffn = nn.Sequential(nn.Linear(hidden_size, ff_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_dim, hidden_size))
        self.int_ln1 = nn.LayerNorm(hidden_size)
        self.int_ln2 = nn.LayerNorm(hidden_size)
        self.int_ffn = nn.Sequential(nn.Linear(hidden_size, ff_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_dim, hidden_size))

    def forward(self, emo_seq, int_seq, bottleneck):
        bottleneck_tilde = self.bottleneck_encoder(bottleneck)
        K_prev = bottleneck_tilde.size(1)
        K_l = max(1, K_prev // 2)
        B_l = bottleneck_tilde[:, :K_l, :]
        emo_ctx, _ = self.ca_emo(B_l, emo_seq, emo_seq)
        int_ctx, _ = self.ca_int(B_l, int_seq, int_seq)
        mtca_out = emo_ctx + int_ctx
        bn_hat = self.bn_ln1(B_l + mtca_out)
        bn_ffn = self.bn_ffn(bn_hat)
        bn_up = self.bn_ln2(bn_hat + bn_ffn)
        emo_ca, _ = self.ca_back_emo(emo_seq, bn_up, bn_up)
        emo_hat = self.emo_ln1(emo_seq + emo_ca)
        emo_ffn = self.emo_ffn(emo_hat)
        emo_out = self.emo_ln2(emo_hat + emo_ffn)
        int_ca, _ = self.ca_back_int(int_seq, bn_up, bn_up)
        int_hat = self.int_ln1(int_seq + int_ca)
        int_ffn = self.int_ffn(int_hat)
        int_out = self.int_ln2(int_hat + int_ffn)
        return (emo_out, int_out, bn_up)

class HTBIBlock(nn.Module):

    def __init__(self, hidden_size, num_layers, init_tokens, num_heads, ff_dim, dropout):
        super().__init__()
        assert num_layers >= 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_tokens = init_tokens
        self.bottleneck_init = nn.Parameter(torch.randn(init_tokens, hidden_size))
        self.layers = nn.ModuleList([HTBILayer(hidden_size=hidden_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)])

    def forward(self, emo_seq, int_seq):
        B, _, H = emo_seq.size()
        bottleneck = self.bottleneck_init.unsqueeze(0).expand(B, self.init_tokens, H)
        for layer in self.layers:
            emo_seq, int_seq, bottleneck = layer(emo_seq, int_seq, bottleneck)
        return (emo_seq, int_seq)
