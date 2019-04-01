__Note:- Work in Progress__

# Implementation of Semantic Image Synthesis with Spatially-Adaptive Normalization (SPADE) [PyTorch]
PyTorch unofficial implementation of Semantic Image Synthesis with Spatially-Adaptive Normalization paper by Nvidia Research.

### [project page](https://nvlabs.github.io/SPADE/) |   [paper](https://arxiv.org/abs/1903.07291) | [GTC 2019 demo](https://youtu.be/p5U4NgVGAwg) | [Youtube](https://youtu.be/MXWm6w4E5q0)

## Overview

<img src="https://nvlabs.github.io/SPADE/images/method.png" width="97%">

In many common normalization techniques such as Batch Normalization (<a href="[https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)"><span style="font-weight:normal">Ioffe et al., 2015</span></a>), there are learned affine layers (as in <a href="[https://pytorch.org/docs/stable/nn.html?highlight=batchnorm2d#torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/nn.html?highlight=batchnorm2d#torch.nn.BatchNorm2d)"><span style="font-weight:normal">PyTorch</span></a> and <a href="[https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)"><span style="font-weight:normal">TensorFlow</span></a>) that are applied after the actual normalization step. In SPADE, the affine layer is <i>learned from semantic segmentation map</i>. This is similar to Conditional Normalization (<a href="[https://arxiv.org/abs/1707.00683](https://arxiv.org/abs/1707.00683)"><span style="font-weight:normal">De Vries et al., 2017</span></a> and <a href="[https://arxiv.org/abs/1610.07629](https://arxiv.org/abs/1610.07629)"><span style="font-weight:normal">Dumoulin et al., 2016</span></a>), except that the learned affine parameters now need to be spatially-adaptive, which means we will use different scaling and bias for each semantic label. Using this simple method, semantic signal can act on all layer outputs, unaffected by the normalization process which may lose such information. Moreover, because the semantic information is provided via SPADE layers, random latent vector may be used as input to the network, which can be used to manipulate the style of the generated images.

# Overview of Repo
All the code for the repo can be found in the src-folder.

```python
└── src
    ├── args.py
    ├── dataloader
    │   ├── cityscapes.py
    ├── models
    │   ├── discriminator.py
    │   ├── encoder.py
    │   ├── generator.py
    │   ├── spade.py
    │   └── spade_resblk.py
    └── notebooks
        ├── develpment.ipynb
        └── train_model.ipynb
```

models folder contains the model definitions for all the models discussed in the paper (spade, spade_resblk, generator, encoder, discriminator). The models are built to resemble that discussed in the paper. The source code is easy to read and modify.

dataloader folder contains the dataloader for cityscapes dataset. The code has been copied from [pytorch-segmap](https://github.com/meetshah1995/pytorch-semseg/tree/master/ptsemseg/loader). pytorch0-segmap contains dataloaders for various segmentation tasks. Although, the code is bit old but some tweaks and quick google serach can get you up with the deprecated functions.

notebooks folder contains two notebooks.
* development -> It was the one I used when I started working on the project. I used this notebook for model development. Do not follow this notebook right now, as the models are wrong in it (for the correct implementation use the .py scripts in `models` folder). I will update the notebook with comments and correct model implementaions.
* train_model -> I have 4GB gpu, so I use this notebook to create complete SPADE model with modified generator and discriminator. Also, I reduce the filter size in `spade` block to 64. I am working on this notebook right now and the training process, would be documneted in this. But if you want to make SPADE model, then it provides a quick tutorial on how to use the repo.

## Next things to do
1. Complete the train script, with loss definitions from the pix2pixhd paper
2. Train for some epochs on CityScapes dataset.
3. Update README with the results
4. Blog in progress.
