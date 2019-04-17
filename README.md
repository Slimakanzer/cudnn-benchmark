CuDNN Convolution Benchmark
===============

Prerequisites
-------------
* [CUDA Toolkit](https://docs.nvidia.com/cuda/index.html)
* [cuDNN SDK](https://developer.nvidia.com/cudnn)

Building
--------
Export the path to the CUDA Tookit directory:
```shell
$ export CUDA_PATH=/path_to_cuda
```
Export the path to the cuDNN SDK directory:
```shell
$ export CUDNN_PATH=/path_to_cudnn
```
Check `nvcc`'s version and cuDNN library paths to ensure your setup is correct:
```shell
$ $CUDA_PATH/bin/nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Fri_Feb__8_19:08:17_PST_2019
Cuda compilation tools, release 10.1, V10.1.105
```
```shell
$ ls $CUDNN_PATH/include $CUDNN_PATH/lib64

/usr/local/cuda/cuda/include:
cudnn.h

/usr/local/cuda/cuda/lib64:
libcudnn.so  libcudnn.so.7  libcudnn.so.7.5.0  libcudnn_static.a
```
Build the benchmark:
```shell
$ make
```
```shell
$ mkdir -p bin
$ $CUDA_PATH/bin/nvcc -std=c++11 -I $CUDNN_PATH/include -L $CUDNN_PATH/lib64 src/benchmark.cu -o bin/benchmark -lcudnn -lcurand
```

Running the benchmark
---------------------
```shell
$ ./bin/benchmark file_name data_type all_formats operation_mode num_repeats [input_tensor_format output_tensor_format kernel_tensor_format]
```

The benchmark expects the following arguments, in the order listed:

* `file_name`: path to the file with convolution cases ([example](https://github.com/Slimakanzer/cudnn-benchmark/blob/master/conv_example.txt));
* `data_type`: data type used (accepted values are `fp16`, `fp32`, `fp64`);
* `all_formats`: `1` if all input/output/tensor formats should be tested, `0` to run with specific data formats only;
* `operation_mode`: `0` to compute workspace size only, `1` to compute workspace size and calculation speed;
* `num_repeats`: number of repetitions for each convolution algorithm.

If `all_formats` is set to `0`, the following additional arguments must be specified:
* `input_tensor_format`: input tensor data format (accepted values are `NCHW`, `NHWC`, `NCHW_VECT_C`);
* `output_tensor_format`: output tensor data format (accepted values are `NCHW`, `NHWC`, `NCHW_VECT_C`);
* `kernel_tensor_format`: kernel tensor data format (accepted values are `NCHW`, `NHWC`, `NCHW_VECT_C`).

Example with specific data formats:
```shell
$ ./bin/benchmark conv_example.txt out_example.txt fp32 0 0 100 NHWC NHWC NHWC
```
Example with all data formats:
```shell
$ ./bin/benchmark conv_example.txt out_example.txt fp32 1 0 1000
```
Obtaining results
-----------------
Running the benchmark produces a `benchmark_result.txt` file in your working directory.

Example contents for `./bin/benchmark conv_example.txt fp32 0 0 100 NHWC NHWC NHWC`:

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
```
