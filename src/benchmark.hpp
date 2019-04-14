//
// Created by slimakanzer on 29.03.19.
//

#ifndef BENCHMARK_BENCHMARK_H
#define BENCHMARK_BENCHMARK_H 1

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

struct benchmarkRow {
    int w, h, c, n, k, s, r, pad_w, pad_h, stride_w, stride_h, out_w, out_h, input_stride_w, input_stride_h, filter_stride_w, filter_stride_h;
    cudnnTensorFormat_t inputTensorFormat = CUDNN_TENSOR_NCHW_VECT_C,
            outputTensorFormat = CUDNN_TENSOR_NCHW_VECT_C,
            filterFormat = CUDNN_TENSOR_NCHW_VECT_C;
};

struct benchmarkResult {
    double time;
    size_t workspace_size;
    benchmarkStatus status;
};

struct benchmarkFwdResult {
    cudnnConvolutionFwdAlgo_t algo;
    benchmarkResult* result;
};

struct benchmarkBwdFilterResult {
    cudnnConvolutionBwdFilterAlgo_t algo;
    benchmarkResult* result;
};

struct benchmarkBwdDataResult {
    cudnnConvolutionBwdDataAlgo_t algo;
    benchmarkResult* result;
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
    benchmarkOperationMode operation_mode;

    TensorDescriptor *inputTensorDescriptor;
    TensorDescriptor *outputTensorDescriptor;
    FilterDescriptor *filterDescriptor;
    cudnnConvolutionDescriptor_t convolutionDescriptor_;
    curandGenerator_t curand_gen;

    size_t fwd_workspace_size(cudnnConvolutionFwdAlgo_t algo);

    size_t bwd_filter_workspace_size(cudnnConvolutionBwdFilterAlgo_t algo);

    size_t bwd_data_workspace_size(cudnnConvolutionBwdDataAlgo_t algo);

    void forward(cudnnConvolutionFwdAlgo_t algo, uint32_t num_repeats);

    void backward_filter(cudnnConvolutionBwdFilterAlgo_t algo, uint32_t num_repeats);

    void backward_data(cudnnConvolutionBwdDataAlgo_t algo, uint32_t num_repeats);

    void forward_workspace(cudnnConvolutionFwdAlgo_t algo);

    void backward_filter_workspace(cudnnConvolutionBwdFilterAlgo_t algo);

    void backward_data_workspace(cudnnConvolutionBwdDataAlgo_t);

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
    benchmarkRow *benchmark_row;
    std::vector<benchmarkFwdResult> fwd_result;
    std::vector<benchmarkBwdFilterResult> bwd_filter_result;
    std::vector<benchmarkBwdDataResult> bwd_data_result;

    Benchmark(benchmarkOperationMode operation_mode);

    void benchmark(benchmarkRow &benchmarkInput, uint32_t num_repeats);

    static void run(std::string file_name, bool all_formats, benchmarkOperationMode operation_mode, uint32_t num_repeats, cudnnTensorFormat_t input_format, cudnnTensorFormat_t output_format, cudnnTensorFormat_t kernel_format);
};

#endif //BENCHMARK_BENCHMARK_H
