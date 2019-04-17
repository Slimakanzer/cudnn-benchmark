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
* `output_file_name`: path to the output file with benchmark results;
* `data_type`: data type used (accepted values are `fp16`, `fp32`, `fp64`);
* `all_formats`: `1` if all input/output/tensor formats should be tested, `0` to run with specific data formats only;
* `operation_mode`: one of the operation modes (see below);
* `num_repeats`: number of repetitions for each convolution algorithm.

If `all_formats` is set to `0`, the following additional arguments must be specified:
* `input_tensor_format`: input tensor data format (accepted values are `NCHW`, `NHWC`, `NCHW_VECT_C`);
* `output_tensor_format`: output tensor data format (accepted values are `NCHW`, `NHWC`, `NCHW_VECT_C`);
* `kernel_tensor_format`: kernel tensor data format (accepted values are `NCHW`, `NHWC`, `NCHW_VECT_C`).

### Operation modes
* `0`: measure workspace memory size only;
* `1`: measure execution time and workspace size.

### Examples

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
Running the benchmark produces a `output_file_name` file in your working directory.

Example contents for `./bin/benchmark conv_example.txt out_example.txt fp32 0 1 10 NHWC NHWC NHWC`:

```
input_format	output_format	filter_format	W	H	C	N	K	S	R	pad_w	pad_h	stride_w	stride_h	out_w	out_h	input_stride_w	input_stride_h	filter_stride_w	filter_stride_h	FWD_GEMM	FWD_GEMM WORKSPACE	FWD_IMPLICIT_GEMM	FWD_IMPLICIT_GEMM WORKSPACE	FWD_PRECOMP_GEMM	FWD_PRECOMP_GEMM WORKSPACE	FWD_DIRECT	FWD_DIRECT WROKSPACE	FWD_FFT	FWD_FFT WORKSPACE	FWD_FFT_TILING	FWD_FFT_TILING WORKSPACE	FWD_WINOGRAD	FWD_WINOGRAD WORKSPACE	FWD_WINOGRAD_NONFUSED	FWD_WINOGRAD_NONFUSED WORKSPACE	BWD_FILTER_ALGO_0	BWD_FILTER_ALGO_0 WORKPACE	BWD_FILTER_ALGO_1	BWD_FILTER_ALGO_1 WORKSPACE	BWD_FILTER_ALGO_3	BWD_FILTER_ALGO_3 WORKSPACE	BWD_FILTER_FFT	BWD_FILTER_FFT WORKSPACE	BWD FILTER FFT_TILING	BWD FILTER FFT_TILING WORKSPACE	BWD_DATA_ALGO_0	BWD_DATA_ALGO_0 WORKSPACE	BWD_DATA_ALGO_1	BWD_DATA_ALGO_1 WORKSPACE	BWD_DATA_FFT	BWD_DATA_FFT WORKSPACE	BWD_DATA_FFT_TILING	BWD_DATA_FFT_TILING WORKSPACE	BWD_DATA_WINOGRAD	BWD_DATA_WINOGRAD WORKSPACE	BWD_DATA_WINOGRAD_NONFUSED	BWD_DATA_WINOGRAD_NONFUSED WORKSPACE
NHWC	NHWC	NHWC	56	56	256	16	64	1	1	0	0	1	1	56	56	1	1	1	1	n/a		8794.02	0	3129.28	18824	n/a		n/a		n/a		n/a		n/a		5353.65	0	6889.87	128	n/a		n/a		n/a		n/a		8355.89	0	n/a		n/a		n/a		n/a		
NHWC	NHWC	NHWC	56	56	256	32	64	1	1	0	0	1	1	56	56	1	1	1	1	n/a		19531.8	0	5817.58	18824	n/a		n/a		n/a		n/a		n/a		9987.11	0	12251.6	128	n/a		n/a		n/a		n/a		16486	0	n/a		n/a		n/a		n/a		
NHWC	NHWC	NHWC	56	56	256	64	64	1	1	0	0	1	1	56	56	1	1	1	1	n/a		39313.3	0	11589.9	18824	n/a		n/a		n/a		n/a		n/a		20037.5	0	23890	128	n/a		n/a		n/a		n/a		-		n/a		n/a		n/a		n/a		
NHWC	NHWC	NHWC	56	56	256	16	128	1	1	0	0	1	1	56	56	1	1	1	1	n/a		9658.47	0	5408.23	18824	n/a		n/a		n/a		n/a		n/a		8656.95	0	9161.4	256	n/a		n/a		n/a		n/a		12674.3	0	n/a		n/a		n/a		n/a		
NHWC	NHWC	NHWC	56	56	256	32	128	1	1	0	0	1	1	56	56	1	1	1	1	n/a		18395.2	0	10729.8	18824	n/a		n/a		n/a		n/a		n/a		17171	0	17812	256	n/a		n/a		n/a		n/a		25342.1	0	n/a		n/a		n/a		n/a		
NHWC	NHWC	NHWC	56	56	256	64	128	1	1	0	0	1	1	56	56	1	1	1	1	n/a		36282.8	0	21465.5	18824	n/a		n/a		n/a		n/a		n/a		34545.3	0	35549.7	256	n/a		n/a		n/a		n/a		-		n/a		n/a		n/a		n/a		
.....
```
