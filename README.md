CuDNN Convolution Benchmark
===============

BUILDING
--------
Setup CUDA_PATH where your [Cuda Toolkit](https://docs.nvidia.com/cuda/index.html) downloaded
```shell
$ export CUDA_PATH=/path_to_cuda
```
Setup CUDNN_PATH where your [cuDNN developer lib](https://developer.nvidia.com/cudnn) downloaded
```shell
$ export CUDNN_PATH=/path_to_cudnn
```

Please make sure CUDA_PATH and CUDNN_PATH environment variables are valid:
```shell
$ $CUDA_PATH/bin/nvcc --version
```
Result:
```shell
$ nvcc: NVIDIA (R) Cuda compiler driver
$ Copyright (c) 2005-2019 NVIDIA Corporation
$ Built on Fri_Feb__8_19:08:17_PST_2019
$ Cuda compilation tools, release 10.1, V10.1.105
```
```shell
$ ls $CUDNN_PATH/include $CUDNN_PATH/lib64
```
Result:
```shell
$ /usr/local/cuda/cuda/include:
$ cudnn.h
$
$ /usr/local/cuda/cuda/lib64:
$ libcudnn.so  libcudnn.so.7  libcudnn.so.7.5.0  libcudnn_static.a
```
Build benchmark:
```shell
$ make
```
