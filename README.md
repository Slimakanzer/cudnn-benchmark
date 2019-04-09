CuDNN Convolution Benchmark
===============

Building
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
```shell
$ nvcc: NVIDIA (R) Cuda compiler driver
$ Copyright (c) 2005-2019 NVIDIA Corporation
$ Built on Fri_Feb__8_19:08:17_PST_2019
$ Cuda compilation tools, release 10.1, V10.1.105
```

```shell
$ ls $CUDNN_PATH/include $CUDNN_PATH/lib64
```
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
```shell
$ mkdir -p bin
$/usr/local/cuda/bin/nvcc -std=c++11 -I /usr/local/cuda/cuda/include -L /usr/local/cuda/cuda/lib64 src/benchmark.cu -o bin/benchmark \
$-lcudnn -lcurand
```

Starting
--------
Benchmark have 9 command line arguments.

    file_name - name of file with convolution cases [example](https://github.com/Slimakanzer/cudnn-benchmark/blob/master/conv_example.txt)
    data_type - data types 
        valuses: fp16, fp32, fp64
    all_formats - use all formats or not
        valuse: 0, 1
    only_workspace - meter only workspace size or not
        values: 0, 1
    num_repeats - number of repeats for one convolution's algorithm
    input_tensor_format - input tensor data format (only if you set 0 for all_formats)
        values: NCHW, NHWC, NCHW_VECT_C
    output_tensor_format - input tensor data format (only if you set 0 for all_formats)
        values: NCHW, NHWC, NCHW_VECT_C
    kernel_tensor_format - input tensor data format (only if you set 0 for all_formats)
        values: NCHW, NHWC, NCHW_VECT_C

Example with input formats:
```shell
$ ./bin/benchmark conv_example.txt fp32 0 0 100 NHWC NHWC NHWC
```
Example with all formats:
```shell
$ ./bin/benchmark conv_example.txt fp32 1 0 1000
```
#Result:
