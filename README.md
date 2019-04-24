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

[Example](https://github.com/Slimakanzer/cudnn-benchmark/blob/master/out_example.txt) contents for `./bin/benchmark conv_example.txt out_example.txt fp32 0 1 10 NCHW NCHW NCHW`:

A value `n/a` means that the combination of the input tensor dimension, filter tensor dimension and output tensor dimension is not supported for the specified algorithm on your GPU.

A value `-` means that this convolution not supported for the specified algorithm on your GPU.

```
input_format	output_format	filter_format	W	H	C	N	K	S	R	pad_w	pad_h	stride_w	stride_h	out_w	out_h	input_stride_w	input_stride_h	filter_stride_w	filter_stride_h	FWD_GEMM	FWD_GEMM WORKSPACE	FWD_IMPLICIT_GEMM	FWD_IMPLICIT_GEMM WORKSPACE	FWD_PRECOMP_GEMM	FWD_PRECOMP_GEMM WORKSPACE	FWD_DIRECT	FWD_DIRECT WROKSPACE	FWD_FFT	FWD_FFT WORKSPACE	FWD_FFT_TILING	FWD_FFT_TILING WORKSPACE	FWD_WINOGRAD	FWD_WINOGRAD WORKSPACE	FWD_WINOGRAD_NONFUSED	FWD_WINOGRAD_NONFUSED WORKSPACE	BWD_FILTER_ALGO_0	BWD_FILTER_ALGO_0 WORKPACE	BWD_FILTER_ALGO_1	BWD_FILTER_ALGO_1 WORKSPACE	BWD_FILTER_ALGO_3	BWD_FILTER_ALGO_3 WORKSPACE	BWD_FILTER_FFT	BWD_FILTER_FFT WORKSPACE	BWD FILTER FFT_TILING	BWD FILTER FFT_TILING WORKSPACE	BWD_DATA_ALGO_0	BWD_DATA_ALGO_0 WORKSPACE	BWD_DATA_ALGO_1	BWD_DATA_ALGO_1 WORKSPACE	BWD_DATA_FFT	BWD_DATA_FFT WORKSPACE	BWD_DATA_FFT_TILING	BWD_DATA_FFT_TILING WORKSPACE	BWD_DATA_WINOGRAD	BWD_DATA_WINOGRAD WORKSPACE	BWD_DATA_WINOGRAD_NONFUSED	BWD_DATA_WINOGRAD_NONFUSED WORKSPACE
NCHW	NCHW	NCHW	1	1	256	32	324	3	3	1	1	1	1	1	1	1	1	1	1	370.914	294912	471.803	0	587.052	9216	n/a		15303.9	212963328	42105.4	441769984	7298.59	8754448	2435.43	14616576	8761.88	0	333.403	6336	8901.35	0	n/a		n/a		3661.55	0	1157.1	0	11922.6	217976832	38886.3	441769984	6682.76	8360960	2169.12	14616576	
NCHW	NCHW	NCHW	1	1	256	32	16	3	3	1	1	1	1	1	1	1	1	1	1	231.894	294912	457.18	0	452.277	9216	n/a		2915.1	28459008	9094.39	55730176	562.581	671808	417.447	1843200	369.539	0	43.1993	576	366.571	0	n/a		n/a		494.008	0	291.386	0	1501.37	19021824	4851.48	55730176	649.183	410624	398.059	1843200	
NCHW	NCHW	NCHW	3	3	256	32	324	3	3	1	1	1	1	3	3	1	1	1	1	1474.53	2654208	2193.58	0	1353.02	60	n/a		15177.2	212963328	43215.7	441769984	3892.56	8754448	2453.19	14616576	9273.82	0	1711.97	2572	1770.54	2356	13231.9	191476224	n/a		4373.88	0	1019.39	2236	11924.6	217976832	39019.9	441769984	3586.94	8360960	2173.33	14616576	
NCHW	NCHW	NCHW	3	3	256	32	16	3	3	1	1	1	1	3	3	1	1	1	1	348.08	2654208	652.083	0	301.795	60	n/a		2989.4	28459008	9137.38	55730176	421.972	671808	420.687	1843200	423.354	0	139.508	2428	126.237	2356	1967.64	20072448	n/a		965.804	0	123.734	2236	1342.91	19021824	4955.75	55730176	352.066	410624	411.616	1843200	
.....
```
