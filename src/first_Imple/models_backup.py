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
        # x : (B, K, t)
        res = x
        h = self.conv1(x)          # (B, K, t)
        h = self.relu(h)
        h = self.conv2(h)          # (B, K, t)
        out = self.relu(h + res)
        return out


# prend en input un batch de sous séquences (B,C,t) avec t = % T
# Ressort un (B,K,t)
class Encoder(nn.Module):
    def __init__(self, in_channel=1, representation_dim=8, num_blocks=10, kernel_size=3):
        super().__init__()

        self.representation_dim = representation_dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.in_channel = in_channel

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

    def forward(self, x):
        # x : (B,C,t) C=1 en univarié
        z = self.input_proj_layer(x)
        # z : (B,K,t) avec K dimension de représentation
        r = self.blocks(z) # (B,K,t)
        return r