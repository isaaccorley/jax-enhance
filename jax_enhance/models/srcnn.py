import jax.numpy as jnp
import flax.linen as nn

from jax_enhance.layers import Sequential, Upsample


class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network https://arxiv.org/pdf/1501.00092v3.pdf"""
    scale_factor: int
    channels: int = 3

    def setup(self):
        relu = lambda x: nn.relu(x)
        self.upsample = Upsample(scale_factor=self.scale_factor, mode="bicubic")
        self.layers = Sequential([
            nn.Conv(features=64, kernel_size=(9, 9)),
            relu,
            nn.Conv(features=32, kernel_size=(1, 1)),
            relu,
            nn.Conv(features=self.channels, kernel_size=(5, 5))
        ])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.upsample(x)
        x = self.layers(x)
        return x
