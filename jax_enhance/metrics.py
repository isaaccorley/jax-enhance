import jax
import jax.numpy as jnp


__all__ = ["mse", "mae", "psnr"]


@jax.jit
def mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(targets - predictions))

@jax.jit
def mae(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(targets - predictions))

@jax.jit
def psnr(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return 10 * jnp.log10(1 / mse(predictions, targets))
