## jax-enhance

--------------------------------------------------------------------------------
jax-enhance is a jax implementation of the [pytorch-enhance library](https://github.com/isaaccorley/pytorch-enhance). This is mostly for my own education/experimentation with jax however you may find these implementations useful.

## Installation
```
pip install git+https://github.com/isaaccorley/jax-enhance
```

## Models
The following models are currently implemented:

* **SRCNN** from Dong et. al [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092v3.pdf)
* **VDSR** from Lee et al. [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/pdf/1511.04587.pdf)
* **ESPCN** from Shi et. al [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158v2.pdf)
* **SRResNet** from Ledig et. al [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802v5.pdf)
* **EDSR** from Lim et. al [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921v1.pdf)

```python
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import jax_enhance

# increase resolution by factor of 2 (e.g. 128x128 -> 256x256)
model = jax_enhance.models.SRResNet(scale_factor=2, channels=3)

lr = jnp.ones(1, 128, 128, 3)
params = model.init(PRNGKey(0), lr)
sr = model.apply(params, lr) #[1, 256, 256, 3]
```

## State-of-the-Art
Not sure which models are currently the best? Check out the [PapersWithCode Image Super-Resolution Leaderboards](https://paperswithcode.com/task/image-super-resolution)

## Losses

* **Perceptual Loss (VGG16)**
* **L1 Loss**
* **L2 Loss**
* **Peak-Signal-Noise-Ratio (PSNR) Loss**

## Metrics

* **Mean Squared Error (MSE)**
* **Mean Absolute Error (MAE)**
* **PSNR**

## Other layers not available in Flax

* **PixelShuffle**
* **Upsample**
* **Sequential**
* **PReLU**
