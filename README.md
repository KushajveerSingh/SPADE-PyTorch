__Note:- Work in Progress__

# Implementation of Semantic Image Synthesis with Spatially-Adaptive Normalization
PyTorch unofficial implementation of Semantic Image Synthesis with Spatially-Adaptive Normalization paper by Nvidia Research.

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