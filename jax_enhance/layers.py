from functools import partial
from typing import Sequence, Callable

import jax
import einops
import jax.numpy as jnp
import flax.linen as nn


__all__ = ["Upsample", "PixelShuffle", "Sequential"]


class Upsample(nn.Module):
    scale_factor: int
    mode: str

    def setup(self):
        self.layer = partial(jax.image.resize, method=self.mode)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        bs, h, w, c = x.shape
        x = self.layer(x, shape=(bs, h*self.scale_factor, w*self.scale_factor, c))
        return x


class PixelShuffle(nn.Module):
    scale_factor: int
    channels: int

    def setup(self):
        self.layer = partial(
            einops.rearrange,
            pattern="b h w (h2 w2 c) -> b (h h2) (w w2) c",
            h2=self.scale_factor,
            w2=self.scale_factor
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layer(x)


class Sequential(nn.Module):
    layers: Sequence[Callable]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x
