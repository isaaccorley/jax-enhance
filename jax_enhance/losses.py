import jax.numpy as jnp
import flax.linen as nn


__all__ = [
    "l1_loss", "l2_loss", "psnr_loss",
    "L1Loss", "L2Loss", "PSNRLoss", "VGGLoss"
]


def l1_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(targets - predictions))

def l2_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(targets - predictions))

def psnr_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return 10 * jnp.log10(1 / l2_loss(predictions, targets))


class L1Loss(object):

    def __call__(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        return l1_loss(predictions, targets)


class L2Loss(object):

    def __call__(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        return l2_loss(predictions, targets)


class PSNRLoss(object):

    def __call__(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        return psnr_loss(predictions, targets)


class VGGLoss(object):
    index: str

    def setup(self):
        pass

    def __call__(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        pass
