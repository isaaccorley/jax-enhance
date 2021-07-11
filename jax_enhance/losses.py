import jax.numpy as jnp
import flax.linen as nn

from jax_enhance.metrics import mae, mse, psnr


__all__ = ["L1Loss", "L2Loss", "PSNRLoss", "VGGLoss"]


class L1Loss(nn.Module):

    def __call__(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        return mae(y_pred, y_true)


class L2Loss(nn.Module):

    def __call__(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        return mse(y_pred, y_true)


class PSNRLoss(nn.Module):

    def __call__(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        return psnr(y_pred, y_true)


class VGGLoss(nn.Module):
    index: str

    def setup(self):
        pass

    def __call__(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        pass
