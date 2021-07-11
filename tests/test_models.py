import itertools

import pytest
import jax
import jax.numpy as jnp

from jax_enhance import models


IMAGE_SIZE = 32
SCALE_FACTOR = [2, 3, 4]
CHANNELS = [1, 3]
BATCH_SIZE = [1, 2]
MODELS = [models.Bicubic, models.SRCNN, models.VDSR, models.ESPCN]
params = list(itertools.product(MODELS, SCALE_FACTOR, CHANNELS, BATCH_SIZE))


@pytest.mark.parametrize("module, scale_factor, channels, batch_size", params)
def test_models(module, scale_factor, channels, batch_size):
    model = module(scale_factor, channels)
    lr = jnp.ones((batch_size, IMAGE_SIZE, IMAGE_SIZE, channels))
    params = model.init(jax.random.PRNGKey(0), lr)
    sr = model.apply(params, lr)
    assert sr.shape == (batch_size, IMAGE_SIZE*scale_factor, IMAGE_SIZE*scale_factor, channels)