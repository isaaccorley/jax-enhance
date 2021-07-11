import jax.numpy as jnp
import flax.linen as nn

from jax_enhance.layers import Sequential, PixelShuffle


class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network https://arxiv.org/pdf/1609.05158v2.pdf"""
    scale_factor: int
    channels: int = 3

    def setup(self):
        relu = lambda x: nn.relu(x)
        self.layers = Sequential([
            nn.Conv(features=64, kernel_size=(5, 5)),
            relu,
            nn.Conv(features=64, kernel_size=(3, 3)),
            relu,
            nn.Conv(features=32, kernel_size=(3, 3)),
            relu,
            nn.Conv(features=self.channels * self.scale_factor ** 2, kernel_size=(3, 3)),
        ])
        self.upsample = PixelShuffle(self.scale_factor)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.layers(x)
        x = self.upsample(x)
        return x
