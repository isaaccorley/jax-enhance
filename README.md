![](assets/pytorch-enhance-logo-cropped.png)

# pytorch-enhance: Image Super-Resolution in PyTorch
[![PyPI version](https://badge.fury.io/py/torch-enhance.svg)](https://badge.fury.io/py/torch-enhance)
![PyPI - Downloads](https://img.shields.io/pypi/dm/torch-enhance?style=plastic)
![GitHub](https://img.shields.io/github/license/IsaacCorley/pytorch-enhance?style=plastic)
![Travis (.com)](https://img.shields.io/travis/com/IsaacCorley/pytorch-enhance?style=plastic)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3739368.svg)](https://doi.org/10.5281/zenodo.3739368)

Library for Minimal Modern Image Super-Resolution in PyTorch


--------------------------------------------------------------------------------
PyTorch Enhance provides a consolidated package of popular Image Super-Resolution models, datasets, and metrics to allow for quick and painless benchmarking or for quickly adding pretrained models to your application.

## Documentation

[https://pytorch-enhance.readthedocs.io](https://pytorch-enhance.readthedocs.io)

## Installation

### pip
```
pip install torch-enhance
```

### latest
```
git clone https://github.com/IsaacCorley/pytorch-enhance.git
cd pytorch-enhance
python setup.py install
```

## Models
The following models are currently implemented:

* **SRCNN** from Dong et. al [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092v3.pdf)
* **VDSR** from Lee et al. [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/pdf/1511.04587.pdf)
* **ESPCN** from Shi et. al [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158v2.pdf)
* **SRResNet** from Ledig et. al [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802v5.pdf)
* **EDSR** from Lim et. al [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921v1.pdf)

```python
import torch
import torch_enhance

# increase resolution by factor of 2 (e.g. 128x128 -> 256x256)
model = torch_enhance.models.SRResNet(scale_factor=2, channels=3)

lr = torch.randn(1, 3, 128, 128)
sr = model(x) # [1, 3, 256, 256]
```

## State-of-the-Art
Not sure which models are currently the best? Check out the [PapersWithCode Image Super-Resolution Leaderboards](https://paperswithcode.com/task/image-super-resolution)


## Datasets
The following benchmark datasets are available for usage:

* **[BSDS100](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU)**
* **[BSDS200](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU)**
* **[BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)**
* **[BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)**
* **[Set5](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU)**
* **[Set14](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU)**
* **[T91](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU)**
* **[Historical](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU)**
* **[Urban100](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU)**
* **[Manga109](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU)**
* **[General100](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU)**
* **[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)**


## Dataset Samples

**BSDS300**                 |  **BSDS500**              |   **T91**
:-------------------------:|:-------------------------:|:-------------------------:
![](assets/BSDS300.gif)  |  ![](assets/BSDS500.gif)     | ![](assets/T91.gif) 

**Set5**                    |  **Set14**                |   **Historical**
:-------------------------:|:-------------------------:|:-------------------------:
![](assets/Set5.gif)  |  ![](assets/Set14.gif)          | ![](assets/Historical.gif) 

## Losses

* **Perceptual Loss (VGG16)**

## Metrics

* **Mean Squared Error (MSE)**
* **Mean Absolute Error (MAE)**
* **Peak-Signal-Noise-Ratio (PSNR)**

## Examples

```
$ cd examples
```

* **[Get up and benchmarking quickly with PyTorch Lightning](examples/pytorch_lightning_example.py)**
* **[Coming from Keras? Try our example using the Poutyne library](examples/poutyne_example.py)**

## Running Tests

```
$ pytest -ra
```

## Cite

Please cite this repository if you used this code in your own work:

```
@software{isaac_corley_2020_3739368,
  author       = {Isaac Corley},
  title        = {PyTorch Enhance},
  month        = apr,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {0.1.2},
  doi          = {10.5281/zenodo.3739368},
  url          = {https://doi.org/10.5281/zenodo.3739368}
}
```
