from functools import partial
from typing import Any, Sequence, Callable

import jax
import einops
import jax.numpy as jnp
import flax.linen as nn


__all__ = ["constant", "Upsample", "PixelShuffle", "PReLU", "Sequential"]


def constant(key, shape: Sequence[int], value: Any, dtype: Any = jnp.float32) -> jnp.ndarray:
    value = jnp.asarray(value, dtype)
    return jnp.ones(shape, dtype) * value


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

    def setup(self):
        self.layer = partial(
            einops.rearrange,
            pattern="b h w (h2 w2 c) -> b (h h2) (w w2) c",
            h2=self.scale_factor,
            w2=self.scale_factor
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layer(x)


class PReLU(nn.Module):
    negative_slope_init: float = 0.01
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, self.dtype)
        negative_slope = self.param(
            "negative_slope",
            partial(constant, value=self.negative_slope_init, dtype=self.dtype),
            (1,)
        )
        return jnp.where(x >= 0, x, negative_slope * x)


class Sequential(nn.Module):
    layers: Sequence[Callable]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x
