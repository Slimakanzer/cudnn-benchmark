//
// Created by slimakanzer on 29.03.19.
//

#ifndef BENCHMARK_BENCHMARK_H
#define BENCHMARK_BENCHMARK_H

#include "helpers/cuda_helper.h"
#include "helpers/cudnn_helper.h"
#include "tensor.h"

enum benchmarkOperationMode {
    ONLY_WORKSPACE_SIZE_MODE = 0,
    CALCULATION_AND_WORKSPACE_SIZE_MODE = 1,
};

enum benchmarkStatus {
    BENCHMARK_SUCCESS = 0,
    BENCHMARK_NOT_SUPPORTED = 1,
    BENCHMARK_ERROR = 2
};


struct benchmarkResult {
    double time;
    size_t workspace_size;
    benchmarkStatus status;
};

struct benchmarkRow {
    int w, h, c, n, k, s, r, pad_w, pad_h, stride_w, stride_h, out_w, out_h, input_stride_w = 1, input_stride_h = 1, filter_stride_w = 1, filter_stride_h = 1;
    cudnnTensorFormat_t
            inputTensorFormat,
            outputTensorFormat,
            filterFormat;
    benchmarkResult
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,

            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,

            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
};

template<typename T>
class Benchmark {
    cudnnHandle_t cudnn;
    Tensor<T> *inputTensor;
    Tensor<T> *outputTensor;
    Tensor<T> *kernelTensor;
    Tensor<T> *delta;
    Tensor<T> *dW;
    Tensor<T> *dX;
    const float alpha = 1, beta = 0;

    TensorDescriptor *inputTensorDescriptor;
    TensorDescriptor *outputTensorDescriptor;
    FilterDescriptor *filterDescriptor;
    cudnnConvolutionDescriptor_t convolutionDescriptor_;
    curandGenerator_t curand_gen;

    size_t fwd_workspace_size(cudnnConvolutionFwdAlgo_t algo);

    size_t bwd_filter_workspace_size(cudnnConvolutionBwdFilterAlgo_t algo);

    size_t bwd_data_workspace_size(cudnnConvolutionBwdDataAlgo_t algo);

    benchmarkResult forward(cudnnConvolutionFwdAlgo_t algo, uint32_t num_repeats);

    benchmarkResult backward_filter(cudnnConvolutionBwdFilterAlgo_t algo, uint32_t num_repeats);

    benchmarkResult backward_data(cudnnConvolutionBwdDataAlgo_t algo, uint32_t num_repeats);

    benchmarkResult forward_workspace(cudnnConvolutionFwdAlgo_t algo);

    benchmarkResult backward_filter_workspace(cudnnConvolutionBwdFilterAlgo_t algo);

    benchmarkResult backward_data_workspace(cudnnConvolutionBwdDataAlgo_t);

    void forward_algorythms(uint32_t num_repeats);

    void backward_filter_algorythms(uint32_t num_repeats);

    void backward_data_algorythms(uint32_t num_repeats);

    void forward_algorythms_workspace();

    void backward_filter_algorythms_workspace();

    void backward_data_algorythms_workspace();

    void calculate_workspace_benchmark(uint32_t num_repeats);

    void workspace_benchmark();

    void create_cudnn();

    void create_curand_generator();

public:
    benchmarkOperationMode operation_mode;
    benchmarkRow *benchmark_row;

    Benchmark(benchmarkOperationMode operation_mode);

    void benchmark(benchmarkRow &benchmarkInput, uint32_t num_repeats);

    static void run(std::string file_name, std::string output_file_name, bool all_formats, benchmarkOperationMode operation_mode, uint32_t num_repeats, cudnnTensorFormat_t input_format, cudnnTensorFormat_t output_format, cudnnTensorFormat_t kernel_format);
};

#endif //BENCHMARK_BENCHMARK_H
