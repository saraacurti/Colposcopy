import torch
import torch.nn as nn

# ============================================================
# 🔹 Multi-Head Self-Attention con Dropout
# ============================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "Dim deve essere divisibile per num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.fc_out = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.fc_out(out)
        out = self.proj_drop(out)
        return out


# ============================================================
# 🔹 Transformer Block (pre-norm + dropout + residuo)
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout=dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm transformer block
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 🔹 Vision Transformer (ViT) con Dropout e PreNorm
# ============================================================
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=128, depth=3, heads=4, mlp_ratio=2.0,
                 num_classes=3, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "image_size deve essere multiplo di patch_size"
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding con Conv2d
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        # Token CLS e Positional Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        # Stack di Transformer Block
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )

    def forward(self, x):
        B = x.size(0)

        # Patchify: (B, dim, H/p, W/p) → (B, num_patches, dim)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Aggiungi token [CLS]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        for blk in self.blocks:
            x = blk(x)

        # Normalizzazione e classificazione
        x = self.norm(x[:, 0])  # solo CLS token
        return self.head(x)

