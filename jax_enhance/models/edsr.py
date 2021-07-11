import math
from functools import partial
from typing import Sequence, Callable

import jax.numpy as jnp
import flax.linen as nn

from jax_enhance.layers import Sequential, PixelShuffle


class ResidualBlock(nn.Module):
    channels: int
    kernel_size: Sequence[int]
    res_scale: float
    activation: Callable

    def setup(self):
        self.layers = Sequential([
            nn.Conv(features=self.channels, kernel_size=self.kernel_size),
            self.activation,
            nn.Conv(features=self.channels, kernel_size=self.kernel_size),
        ])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.layers(x)


class UpsampleBlock(nn.Module):
    num_upsamples: int
    channels: int
    kernel_size: Sequence[int]

    def setup(self):
        layers = []
        for _ in range(self.num_upsamples):
            layers.extend([
                nn.Conv(features=self.channels * 2 ** 2, kernel_size=self.kernel_size),
                PixelShuffle(scale_factor=2, channels=self.channels),
            ])
        self.layers = Sequential(layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layers(x)


class EDSR(nn.Module):
    """Enhanced Deep Residual Networks for Single Image Super-Resolution https://arxiv.org/pdf/1707.02921v1.pdf"""
    scale_factor: int
    channels: int = 3
    num_blocks: int = 32

    def setup(self):


        relu = lambda x: nn.relu(x)

        # pre res blocks layer
        self.head = nn.Conv(features=256, kernel_size=(3, 3))

        # res blocks
        res_blocks = [
            ResidualBlock(channels=256, kernel_size=(3, 3), res_scale=0.1, activation=relu)
            for i in range(self.num_blocks)
        ]
        res_blocks.append(nn.Conv(features=256, kernel_size=(3, 3)))
        self.res_blocks = Sequential(res_blocks)

        # upsample
        num_upsamples = int(math.log2(self.scale_factor))
        self.upsample = UpsampleBlock(num_upsamples=num_upsamples, channels=256, kernel_size=(3, 3))

        # output layer
        self.tail = nn.Conv(self.channels, kernel_size=(3, 3))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.head(x)
        x = x + self.res_blocks(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x
