import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================
# Multi-Head Self Attention 
# ============================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.fc_out(out)

# ============================
# Transformer Block 
# ============================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ============================
# Vision Transformer
# ============================
class ViT(nn.Module):
    def __init__(self, image_size=128, patch_size=16, dim=128, depth=3, heads=4, num_classes=3):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Solo 3 blocchi 
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # Dividsione immagine patch
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p, p).permute(0, 2, 1, 3, 4)
        x = x.reshape(B, -1, 3 * p * p)
        x = self.patch_embed(x)

        #  CLS e positional embedding --> token classe e codifica di posizione
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # Passa nei blocchi transformer
        for blk in self.blocks:
            x = blk(x)

        # Classificazione
        x = self.norm(x[:, 0])
        return self.head(x)
