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
        values:
            fp16 - use 16 bit float point values
            fp32 - use 32 bit float point values
            fp64 - use 64 bit float point values
    all_formats - use all formats or not
        values: 
            0 - turn off (you need set format names in the next cmd line args.) 
            1 - turn on
    operation_mode - benchmark operation mode
        values:
            0 - compute only workspace size
            1 - compute calculation speed and workspace size
    num_repeats - number of repeats for one convolution's algorithm
    input_tensor_format - input tensor data format (only if you set 0 for all_formats)
        values: 
            [NCHW](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnTensorFormat_t)
            [NHWC](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnTensorFormat_t)
            [NCHW_VECT_C](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnTensorFormat_t)
    output_tensor_format - input tensor data format (only if you set 0 for all_formats)
        values:
            [NCHW](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnTensorFormat_t)
            [NHWC](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnTensorFormat_t)
            [NCHW_VECT_C](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnTensorFormat_t)
    kernel_tensor_format - input tensor data format (only if you set 0 for all_formats)
        values:
            [NCHW](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnTensorFormat_t)
            [NHWC](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnTensorFormat_t)
            [NCHW_VECT_C](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnTensorFormat_t)

Example with input formats:
```shell
$ ./bin/benchmark conv_example.txt out_example.txt fp32 0 0 100 NHWC NHWC NHWC
```
Example with all formats:
```shell
$ ./bin/benchmark conv_example.txt out_example.txt fp32 1 0 1000
```
Result
------
As a result you have file_name_output file where you can see result of benchmark.
```shell
$ ./bin/benchmark conv_example.txt out_example.txt fp32 0 1 10 NHWC NHWC NHWC
```
        	input_format	output_format	filter_format	W	H	C	N	K	S	R	pad_w	pad_h	stride_w	stride_h	out_w	out_h	input_stride_w	input_stride_h	filter_stride_w	filter_stride_h

        	NHWC	NHWC	NHWC	56	56	256	16	64	1	1	0	0	1	1	56	56	1	1	1	1
        FWD_GEMM                    	n/a
        FWD_IMPLICIT_GEMM           	9158.3	0
        FWD_PRECOMP_GEMM            	2919.81	18824
        FWD_DIRECT                  	n/a
        FWD_FFT                     	n/a
        FWD_FFT_TILING              	n/a
        FWD_WINOGRAD                	n/a
        FWD_WINOGRAD_NONFUSED       	n/a
        BWD_FILTER_ALGO_0           	4981.08	0
        BWD_FILTER_ALGO_1           	6205.38	128
        BWD_FILTER_ALGO_3           	n/a
        BWD_FILTER_FFT              	n/a
        BWD FILTER FFT_TILING       	n/a
        BWD_DATA_ALGO_0             	n/a
        BWD_DATA_ALGO_1             	8354.06	0
        BWD_DATA_FFT                	n/a
        BWD_DATA_FFT_TILING         	n/a
        BWD_DATA_WINOGRAD           	n/a
        BWD_DATA_WINOGRAD_NONFUSED  	n/a


        	NHWC	NHWC	NHWC	56	56	256	32	64	1	1	0	0	1	1	56	56	1	1	1	1
        FWD_GEMM                    	n/a
        FWD_IMPLICIT_GEMM           	19308.8	0
        FWD_PRECOMP_GEMM            	5735.22	18824
        FWD_DIRECT                  	n/a
        FWD_FFT                     	n/a
        FWD_FFT_TILING              	n/a
        FWD_WINOGRAD                	n/a
        FWD_WINOGRAD_NONFUSED       	n/a
        BWD_FILTER_ALGO_0           	10174.5	0
        BWD_FILTER_ALGO_1           	12112.3	128
        BWD_FILTER_ALGO_3           	n/a
        BWD_FILTER_FFT              	n/a
        BWD FILTER FFT_TILING       	n/a
        BWD_DATA_ALGO_0             	n/a
        BWD_DATA_ALGO_1             	16180	0
        BWD_DATA_FFT                	n/a
        BWD_DATA_FFT_TILING         	n/a
        BWD_DATA_WINOGRAD           	n/a
        BWD_DATA_WINOGRAD_NONFUSED  	n/a
        .....
