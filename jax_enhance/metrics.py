import jax.numpy as jnp


__all__ = ["mse", "mae", "psnr"]


def mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(y_true - y_pred))

def mae(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(y_true - y_pred))

def psnr(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    return 10 * jnp.log10(1 / mse(y_pred, y_true))
