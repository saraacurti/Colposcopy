# ==========================================
# Vision Transformer implementato da zero
# ==========================================

import torch
import torch.nn as nn

# ============================
# Multi-Head Self-Attention
# ============================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0 
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.fc_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape  # batch, tokens, dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.fc_out(out)
        return self.dropout(out)

# ============================
# Transformer Encoder Block
# ============================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ============================
# Vision Transformer completo
# ============================
class ViTClassifier(nn.Module):
    def __init__(self, image_size=128, patch_size=16, dim=256, depth=6, heads=8, mlp_ratio=4, num_classes=3, dropout=0.1):
        super(ViTClassifier, self).__init__()

        assert image_size % patch_size == 0 
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        # Embedding dei patch
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Stack di encoder transformer
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # Divide immagine in patch e crea embedding
        x = x.unfold(2, p, p).unfold(3, p, p) 
        x = x.contiguous().view(B, C, -1, p, p).permute(0, 2, 1, 3, 4)
        x = x.reshape(B, -1, 3 * p * p)
        x = self.patch_embed(x)

        # Aggiungi il token [CLS]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        # Passa attraverso i blocchi Transformer
        for block in self.blocks:
            x = block(x)

        # Prendi il token [CLS] per la classificazione
        x = self.norm(x[:, 0])
        return self.head(x)
