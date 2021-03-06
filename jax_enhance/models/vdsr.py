from typing import Any

import jax.numpy as jnp
import flax.linen as nn

from jax_enhance.layers import Sequential, Upsample


class VDSR(nn.Module):
    """Very Deep Super Resolution https://arxiv.org/pdf/1511.04587.pdf"""
    scale_factor: int
    channels: int = 3
    num_layers: int = 20
    dtype: Any = jnp.float32

    def setup(self):
        self.upsample = Upsample(scale_factor=self.scale_factor, mode="bicubic")
        layers = []
        for i in range(self.num_layers - 1):
            layers.append(nn.Conv(features=64, kernel_size=(3, 3), dtype=self.dtype))
            layers.append(nn.relu)

        layers.append(nn.Conv(features=self.channels, kernel_size=(3, 3), dtype=self.dtype))
        self.layers = Sequential(layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.upsample(x)
        x = x + self.layers(x)
        return x
