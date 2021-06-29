# pytorch-classification-advprop
A PyTorch implementation of CVPR2020 paper Adversarial examples improve image recognition by Xie C, Tan M, Gong B, et al. 

Thanks for guidance from Cihang Xie and Yingwei Li. The code is adapted from https://github.com/bearpaw/pytorch-classification.

## Environments
This project is developed and tested in the following environments.
* Ubuntu 18.04
* CUDA 10.2
* GTX 1070 TI
* Python 3.8.1

# Training

~~~
python train_from_scarch.py --checkpoint checkpoint/cifar100/resnet50 --dataset cifar100 --gpu-id 3 --mixbn
~~~

- train: training the model 
    tune the parameters: iterations, epsilon,etc.
- net_rectified: ResNet architecture for advprop
- split_bn: split the normal BN and auxiliary BN to two separate Resnet34 model
- compile_to_deepinversion: compile Resnet from "net" to "resnet_cifar"
