import math
from functools import partial
from typing import Any, Sequence, Callable

import jax.numpy as jnp
import flax.linen as nn

from jax_enhance.layers import Sequential, PixelShuffle, PReLU


class ResidualBlock(nn.Module):
    channels: int
    kernel_size: Sequence[int]
    activation: Callable
    norm: Callable
    dtype: Any = jnp.float32

    def setup(self):
        self.layers = Sequential([
            nn.Conv(features=self.channels, kernel_size=self.kernel_size, dtype=self.dtype),
            self.norm(),
            self.activation,
            nn.Conv(features=self.channels, kernel_size=self.kernel_size, dtype=self.dtype),
            self.norm(),
        ])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.layers(x)


class UpsampleBlock(nn.Module):
    num_upsamples: int
    channels: int
    kernel_size: Sequence[int]
    activation: Callable
    dtype: Any = jnp.float32

    def setup(self):
        layers = []
        for _ in range(self.num_upsamples):
            layers.extend([
                nn.Conv(features=self.channels * 2 ** 2, kernel_size=self.kernel_size, dtype=self.dtype),
                PixelShuffle(scale_factor=2),
                self.activation
            ])
        self.layers = Sequential(layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layers(x)


class SRResNet(nn.Module):
    """Super-Resolution Residual Neural Network https://arxiv.org/pdf/1609.04802v5.pdf"""
    scale_factor: int
    channels: int = 3
    num_blocks: int = 16
    dtype: Any = jnp.float32

    def setup(self):
        assert self.scale_factor % 2 == 0, "Scale factor must be divisible by 2"
        norm = partial(nn.BatchNorm, use_running_average=True)

        # pre res blocks layer
        self.head = Sequential([
            nn.Conv(features=64, kernel_size=(9, 9), dtype=self.dtype),
            PReLU()
        ])

        # res blocks
        res_blocks = [
            ResidualBlock(channels=64, kernel_size=(3, 3), activation=PReLU(), norm=norm, dtype=self.dtype)
            for i in range(self.num_blocks)
        ]
        res_blocks.append(nn.Conv(features=64, kernel_size=(3, 3), dtype=self.dtype))
        self.res_blocks = Sequential(res_blocks)

        # upsample
        num_upsamples = int(math.log2(self.scale_factor))
        self.upsample = UpsampleBlock(
            num_upsamples=num_upsamples,
            channels=64,
            kernel_size=(3, 3),
            activation=PReLU(),
            dtype=self.dtype
        )

        # output layer
        self.tail = nn.Conv(self.channels, kernel_size=(9, 9), dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.head(x)
        x = x + self.res_blocks(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x
