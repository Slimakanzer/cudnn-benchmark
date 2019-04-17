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
    file_name_output - name of the output file with benchmark results
    data_type - data types 
        valuses: fp16, fp32, fp64
    all_formats - use all formats or not
        valuse: 0, 1
    operation_mode - benchmark operation mode
        values:
            0 - compute only workspace size
            1 - compute calculation speed and workspace size
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
Result
------
As a result you have benchmark_result.txt file where you can see result of benchmark.
```shell
$ ./bin/benchmark conv_example.txt fp32 0 0 100 NHWC NHWC NHWC
```
        // input_format output_format filter_format W H C N K S R pad_w pad_h stride_w stride_h out_w out_h input_stride_w input_stride_h filter_stride_w filter_stride_h
        // ALGO STATUS TIME WORKSPACE
        
        NHWC NHWC NHWC 35 35 192 64 64 1 1 0 0 1 1 35 35 1 1 1 1
        FWD_GEMM                     n/a
        FWD_IMPLICIT_GEMM            success 11823.3 0
        FWD_PRECOMP_GEMM             success 3606.28 7356
        FWD_DIRECT                   n/a
        FWD_FFT                      n/a
        FWD_FFT_TILING               n/a
        FWD_WINOGRAD                 n/a
        FWD_WINOGRAD_NONFUSED        n/a
        BWD_FILTER_ALGO_0            success 6064.28 0
        BWD_FILTER_ALGO_1            success 7660.23 96
        BWD_FILTER_ALGO_3            n/a
        BWD_FILTER_FFT               n/a
        BWD FILTER FFT_TILING        n/a
        BWD_DATA_ALGO_0              n/a
        BWD_DATA_ALGO_1              success 10893.8 0
        BWD_DATA_FFT                 n/a
        BWD_DATA_FFT_TILING          n/a
        BWD_DATA_WINOGRAD            n/a
        BWD_DATA_WINOGRAD_NONFUSED   n/a


        NHWC NHWC NHWC 35 35 256 32 48 1 1 0 0 1 1 35 35 1 1 1 1
        FWD_GEMM                     n/a
        FWD_IMPLICIT_GEMM            success 6477.93 0
        FWD_PRECOMP_GEMM             success 2225.4 7356
        FWD_DIRECT                   n/a
        FWD_FFT                      n/a
        FWD_FFT_TILING               n/a
        FWD_WINOGRAD                 n/a
        FWD_WINOGRAD_NONFUSED        n/a
        BWD_FILTER_ALGO_0            success 3962.88 0
        BWD_FILTER_ALGO_1            success 4988.61 128
        BWD_FILTER_ALGO_3            n/a
        BWD_FILTER_FFT               n/a
        BWD FILTER FFT_TILING        n/a
        BWD_DATA_ALGO_0              n/a
        BWD_DATA_ALGO_1              success 5675.6 0
        BWD_DATA_FFT                 n/a
        BWD_DATA_FFT_TILING          n/a
        BWD_DATA_WINOGRAD            n/a
        BWD_DATA_WINOGRAD_NONFUSED   n/a
        .....
