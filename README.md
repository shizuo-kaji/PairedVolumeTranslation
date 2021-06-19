Volume-to-volume translation by CNNs trained on paired data
=============
written by Shizuo KAJI

This is an implementation of volume-to-volume translation using a paired volume dataset.
It can be used for various tasks including
- segmentation, denoising, super-resolution, modality-conversion, and reconstruction

The details can be found in our paper:
"Overview of image-to-image translation using deep neural networks: denoising, super-resolution, modality-conversion, and reconstruction in medical imaging"
by Shizuo Kaji and Satoshi Kida, Radiological Physics and Technology,  Volume 12, Issue 3 (2019), pp 235--248,
[arXiv:1905.08603](https://arxiv.org/abs/1905.08603)

If you use this software, please cite the above paper.

## Background
This code is based on 
- https://github.com/pfnet-research/chainer-pix2pix
- https://gist.github.com/crcrpar/6f1bc0937a02001f14d963ca2b86427a

### Requirements
- a modern GPU with a large memory (the crop_size should be reduced for GPUs with less than 20G memory)
- python 3: [Anaconda](https://anaconda.org) is recommended
- chainer >= 7.2.0, cupy, chainerui, chainercv, opecv, pydicom: install them by
```
pip install cupy chainer chainerui chainercv opencv-contrib-python pydicom
```

Note that with GeForce 30 RTX series, 
the installation of chainer and cupy can be a little tricky for now.
You need CUDA >= 11.1 for these GPUs, and it is supported by CuPy >= v8.
The latest version of Chainer v7.7.0 available on pip is not compatible with the latest version of CuPy.
See [here](https://github.com/chainer/chainer/pull/8583).
You can install the latest Chainer directly from the github repository, which is compatible with the latest version of CuPy.
For example, follow the following procedure:
- Install CUDA 11.1
- pip install cupy-cuda111
- pip install -U git+https://github.com/chainer/chainer.git

You will see some warning messages, but you can ignore them.


## Licence
MIT Licence

