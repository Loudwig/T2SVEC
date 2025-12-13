import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channel, dilation, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv1d(
            channel, channel,
            dilation=dilation,
            kernel_size=kernel_size,
            padding="same"
        )
        self.conv2 = nn.Conv1d(
            channel, channel,
            dilation=dilation,
            kernel_size=kernel_size,
            padding="same"
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = (B, K, t)
        res = x
        h = self.conv1(x)
        h = self.relu(h)
        h = self.conv2(h)
        out = self.relu(h +res)
        return out

# prend en input un batch de sous séquences (B,C,t) avec t = % T
# Ressort un (B,K,t)
class Encoder(nn.Module):
    def __init__(
        self,
        in_channel=1,
        representation_dim=8,
        num_blocks=10,
        kernel_size=3,
        mask_prob=0.5,
    ):
        super().__init__()

        self.representation_dim = representation_dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.mask_prob = mask_prob

        # MLP per timestamp = Conv1D
        self.input_proj_layer = nn.Conv1d(in_channel, representation_dim, kernel_size=1)

        # on définit les resBlocks avec dilatation :
        blocks = []
        for i in range(num_blocks):
            bi = ResBlock(
                channel=self.representation_dim,
                dilation=2 ** (i + 1),
                kernel_size=self.kernel_size
            )
            blocks.append(bi)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, apply_mask=True):
        # x : (B, C, T)
        
        # --- 1. Gestion des Missing Values (NaNs) ---
        is_missing = torch.isnan(x).any(dim=1, keepdim=True)
        mask_obs = ~is_missing  # 1 si observé, 0 si manquant

        # Remplacement temporaire des NaNs par 0 dans l'input
        x_filled = torch.nan_to_num(x, nan=0.0)

        # Projection (B, C, T) -> (B, K, T)
        z = self.input_proj_layer(x_filled)
        
        z = z * mask_obs.float()

        
        if self.training and apply_mask and self.mask_prob > 0.0:
            B, K, T = z.shape
            keep_prob = 1.0 - self.mask_prob
            # mask_augment aléatoire pour le contrastive learning
            mask_augment = torch.bernoulli(keep_prob * torch.ones(B, 1, T, device=z.device, dtype=z.dtype))
            z = z * mask_augment
            
        r = self.blocks(z) # (B, K, T)
        return r
