# Invertible DenseNets with Concatenated LipSwish

Code for Invertible DenseNets. 

This work is inspired by [Invertible Residual Networks](https://arxiv.org/abs/1811.00995) and [Residual Flows](https://arxiv.org/abs/1906.02735). The source is adapted from [Residual Flows](https://github.com/rtqichen/residual-flows).

A BibTeX entry for LaTeX users:
```
@misc{perugachidiaz2021invertible,
      title={Invertible DenseNets with Concatenated LipSwish}, 
      author={Yura Perugachi-Diaz and Jakub M. Tomczak and Sandjai Bhulai},
      year={2021},
      eprint={2102.02694}
}
```

## Requirements
* Python (tested with 3.7)
* PyTorch (tested with 1.5.0)

## Download datasets
* CIFAR10 is automatically downloaded. 
* The pre-processing steps and downloading of ImageNet32 are described in [Residual Flows](https://github.com/rtqichen/residual-flows).

## Density estimation
Default settings of i-DenseNets depth and growth are applicable for both CIFAR10 and ImageNet32.

### CIFAR-10 results
```
python train_img.py --data cifar10 --nblocks 16-16-16 --save experiments/cifar10 --densenet True --learnable_concat True --start_learnable_concat 25 --act CLipSwish --densenet_depth 3 --densenet_growth 172
```

#### CIFAR-10 results (smaller architectures)
Code for the smaller architecture:
```
python train_img.py --data cifar10 --nblocks 4-4-4 --save experiments/cifar10_small --densenet True --learnable_concat True --start_learnable_concat 25 --act CLipSwish --densenet_depth ? --densenet_growth ?
```
where ? ? can be replaced with the following depth and growth sizes to utilize a similar number of parameters as the smaller Residual Flow architecture:

* `--densenet_depth 2 --densenet_growth 318`
* `--densenet_depth 3 --densenet_growth 178` (optimal architecture)
* `--densenet_depth 4 --densenet_growth 122`
* `--densenet_depth 5 --densenet_growth 92`

### ImageNet 32x32 results
```
python train_img.py --data imagenet32 --nblocks 32-32-32 --save experiments/imagenet32 --densenet True --learnable_concat True --start_learnable_concat 0 --act CLipSwish --densenet_depth 3 --densenet_growth 172
```
