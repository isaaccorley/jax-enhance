import jax.numpy as jnp
import flax.linen as nn

from jax_enhance.layers import Upsample


class Bicubic(nn.Module):
    scale_factor: int
    channels: int = 3

    def setup(self):
        self.layers = Upsample(scale_factor=self.scale_factor, mode="bicubic")
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layers(x)
