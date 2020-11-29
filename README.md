# latent_3d_points_Pytorch

## About
PyTorch implementation of DGCNN (Deep Graph Convolutional Neural Network). Check https://github.com/optas/latent_3d_points for more information.


## Introduction
This work proposed a novel deep net architecture for auto-encoding point clouds. The learned representations were amenable to semantic part editting, shape analogies, linear classification and shape interpolations.



## Requirements

- python 3.6
- [torch 1.7.0](https://pytorch.org/get-started/locally/)
- CUDA 10.2
- [chrdiller/pyTorchChamferDistance](https://github.com/chrdiller/pyTorchChamferDistance)


## Datasets
- shape_net_core_uniform_samples_2048.zip(run 'data/download_data.sh')



## metric/chamfer_distance
```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
sudo wget -P /usr/bin https://github.com/unlimblue/KNN_CUDA/raw/master/ninja
sudo chmod +x /usr/bin/ninja
```


