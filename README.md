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
* `data_type`: data type used (accepted values are `fp16`, `fp32`, `fp64`, `int8`, `uint8`, `int32`, `int8x4`, `uint8x4`, `uint8x32`);
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

Example contents for `./bin/benchmark conv_example.txt out_example.txt fp32 0 1 10 NCHW NCHW NCHW`:

A value `n/a` means that the combination of the input tensor dimension, filter tensor dimension and output tensor dimension is not supported for the specified algorithm on your GPU.

A value `-` means that this convolution not supported for the specified algorithm on your GPU.
```
input_format	output_format	filter_format	W	H	C	N	K	S	R	pad_w	pad_h	stride_w	stride_h	out_w	out_h	input_stride_w	input_stride_h	filter_stride_w	filter_stride_h	FWD_GEMM	FWD_GEMM WORKSPACE	FWD_IMPLICIT_GEMM	FWD_IMPLICIT_GEMM WORKSPACE	FWD_PRECOMP_GEMM	FWD_PRECOMP_GEMM WORKSPACE	FWD_DIRECT	FWD_DIRECT WROKSPACE	FWD_FFT	FWD_FFT WORKSPACE	FWD_FFT_TILING	FWD_FFT_TILING WORKSPACE	FWD_WINOGRAD	FWD_WINOGRAD WORKSPACE	FWD_WINOGRAD_NONFUSED	FWD_WINOGRAD_NONFUSED WORKSPACE	BWD_FILTER_ALGO_0	BWD_FILTER_ALGO_0 WORKPACE	BWD_FILTER_ALGO_1	BWD_FILTER_ALGO_1 WORKSPACE	BWD_FILTER_ALGO_3	BWD_FILTER_ALGO_3 WORKSPACE	BWD_FILTER_FFT	BWD_FILTER_FFT WORKSPACE	BWD FILTER FFT_TILING	BWD FILTER FFT_TILING WORKSPACE	BWD_DATA_ALGO_0	BWD_DATA_ALGO_0 WORKSPACE	BWD_DATA_ALGO_1	BWD_DATA_ALGO_1 WORKSPACE	BWD_DATA_FFT	BWD_DATA_FFT WORKSPACE	BWD_DATA_FFT_TILING	BWD_DATA_FFT_TILING WORKSPACE	BWD_DATA_WINOGRAD	BWD_DATA_WINOGRAD WORKSPACE	BWD_DATA_WINOGRAD_NONFUSED	BWD_DATA_WINOGRAD_NONFUSED WORKSPACE
NCHW	NCHW	NCHW	56	56	256	16	64	1	1	0	0	1	1	56	56	1	1	1	1	7832.92	51380224	5670.75	0	3090.17	18824	n/a		-	692158464	13932	80216896	n/a		n/a		13630.3	0	9187.77	2856	8499.07	2848	-	566493184	n/a		9851.22	0	5196.68	21000	-	588349440	13253.2	80216896	n/a		n/a		
NCHW	NCHW	NCHW	56	56	256	32	64	1	1	0	0	1	1	56	56	1	1	1	1	16521.9	102760448	11344.2	0	6202.94	18824	n/a		-	830570496	27465.4	158204736	n/a		n/a		25994	0	16950	2856	17832.7	2848	-	579338240	n/a		20215.9	0	10509.1	21000	-	622952448	24148.2	158204736	n/a		n/a		
NCHW	NCHW	NCHW	56	56	256	64	64	1	1	0	0	1	1	56	56	1	1	1	1	-	205520896	25145.7	0	12366.4	18824	n/a		-	1107394560	-	314180416	n/a		n/a		53764.4	0	34309.2	2856	35367.3	2848	-	743440384	n/a		-	0	-	21000	-	692158464	-	314180416	n/a		n/a		
NCHW	NCHW	NCHW	56	56	256	16	128	1	1	0	0	1	1	56	56	1	1	1	1	12200.5	51380224	8295.04	0	5434.24	18824	n/a		-	1245839360	20858.5	98042688	n/a		n/a		11855.1	0	12512.9	2856	12745.3	2848	-	1132986368	n/a		12692.5	0	8008.82	21000	-	1176633344	19378.9	98042688	n/a		n/a		
NCHW	NCHW	NCHW	56	56	256	32	128	1	1	0	0	1	1	56	56	1	1	1	1	23221.5	102760448	16511.4	0	11809.3	18824	n/a		-	1384251392	-	191628096	n/a		n/a		22746.6	0	24933.3	2856	26902.6	2848	-	1158676480	n/a		22377.7	0	16342.6	21000	-	1245839360	-	191628096	n/a		n/a		
NCHW	NCHW	NCHW	56	56	256	64	128	1	1	0	0	1	1	56	56	1	1	1	1	-	205520896	35353.7	0	23024.3	18824	n/a		-	1661075456	-	378798912	n/a		n/a		43519.2	0	51702.1	2856	51937.3	2848	-	1210056704	n/a		-	0	-	21000	-	1384251392	-	378798912	n/a		n/a		
.....
```
