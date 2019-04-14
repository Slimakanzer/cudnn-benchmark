//
// Created by slimakanzer on 29.03.19.
//
#include <assert.h>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include "benchmark.hpp"
#include "parser.hpp"

template<typename T>
void Benchmark<T>::create_cudnn() {
    CHECK_CUDNN_ERROR(cudnnCreate(&cudnn));
}

template<typename T>
void Benchmark<T>::create_curand_generator() {
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);
}

template<typename T>
Benchmark<T>::Benchmark(benchmarkOperationMode operation_mode) {
    create_cudnn();
    create_curand_generator();
    this->operation_mode = operation_mode;
}

template<typename T>
size_t Benchmark<T>::fwd_workspace_size(cudnnConvolutionFwdAlgo_t algo) {
    assert(cudnn);
    assert(inputTensorDescriptor);
    assert(filterDescriptor);
    assert(outputTensorDescriptor);

    size_t workspace_size = 0;
    CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                              inputTensorDescriptor->descriptor(),
                                                              filterDescriptor->descriptor(),
                                                              convolutionDescriptor_,
                                                              outputTensorDescriptor->descriptor(),
                                                              algo,
                                                              &workspace_size));
    return workspace_size;
}

template<typename T>
size_t Benchmark<T>::bwd_filter_workspace_size(cudnnConvolutionBwdFilterAlgo_t algo) {
    assert(cudnn);
    assert(inputTensorDescriptor);
    assert(filterDescriptor);
    assert(outputTensorDescriptor);

    size_t workspace_size = 0;
    CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,
                                                                     inputTensorDescriptor->descriptor(),
                                                                     outputTensorDescriptor->descriptor(),
                                                                     convolutionDescriptor_,
                                                                     filterDescriptor->descriptor(),
                                                                     algo,
                                                                     &workspace_size));
    return workspace_size;
}

template<typename T>
size_t Benchmark<T>::bwd_data_workspace_size(cudnnConvolutionBwdDataAlgo_t algo) {
    assert(cudnn);
    assert(inputTensorDescriptor);
    assert(filterDescriptor);
    assert(outputTensorDescriptor);

    size_t workspace_size = 0;
    CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn,
                                                                   filterDescriptor->descriptor(),
                                                                   outputTensorDescriptor->descriptor(),
                                                                   convolutionDescriptor_,
                                                                   inputTensorDescriptor->descriptor(),
                                                                   algo,
                                                                   &workspace_size));
    return workspace_size;


}

template<typename T>
void Benchmark<T>::forward(cudnnConvolutionFwdAlgo_t algo, uint32_t num_repeats) {
    assert(inputTensor);
    assert(outputTensor);
    assert(kernelTensor);

    size_t workspace_size;
    benchmarkResult *result = (benchmarkResult *) malloc(sizeof(benchmarkResult));
    try {
        workspace_size = fwd_workspace_size(algo);
    } catch (std::exception &exception) {
        *result = {0, 0, BENCHMARK_NOT_SUPPORTED};
        fwd_result.push_back({algo, result});
        return;
    }

    void *d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_size);

    double fwd_time = 0;
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        cudnnStatus_t
                fwd_status = cudnnConvolutionForward(cudnn,
                                                     &alpha,
                                                     inputTensorDescriptor->descriptor(),
                                                     inputTensor->begin(),
                                                     filterDescriptor->descriptor(),
                                                     kernelTensor->begin(),
                                                     convolutionDescriptor_,
                                                     algo,
                                                     d_workspace,
                                                     workspace_size,
                                                     &beta,
                                                     outputTensorDescriptor->descriptor(),
                                                     outputTensor->begin());

        if (fwd_status == CUDNN_STATUS_NOT_SUPPORTED) {
            *result = {0, 0, BENCHMARK_NOT_SUPPORTED};
            fwd_result.push_back({algo, result});
            return;
        } else if (fwd_status != CUDNN_STATUS_SUCCESS) {
            *result = {0, 0, BENCHMARK_ERROR};

            std::cerr << "CUDNN failure: " << cudnnGetErrorString(fwd_status) << std::endl;
            fwd_result.push_back({algo, result});
            return;
        }
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    fwd_time = std::chrono::duration<double, std::micro>(end - start).count() / num_repeats;
    cudaFree(d_workspace);

    *result = {fwd_time, workspace_size, BENCHMARK_SUCCESS};
    fwd_result.push_back({algo, result});
    return;
}

template<typename T>
void Benchmark<T>::backward_filter(cudnnConvolutionBwdFilterAlgo_t algo, uint32_t num_repeats) {
    assert(inputTensor);
    assert(dW);
    assert(delta);

    size_t workspace_size;
    benchmarkResult *result = (benchmarkResult *) malloc(sizeof(benchmarkResult));
    try {
        workspace_size = bwd_filter_workspace_size(algo);
    } catch (std::exception &exception) {
        *result = {0, 0, BENCHMARK_NOT_SUPPORTED};
        bwd_filter_result.push_back({algo, result});
        return;
    }

    void *d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_size);

    double fwd_time = 0;
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        cudnnStatus_t bwd_filter_status = cudnnConvolutionBackwardFilter(cudnn,
                                                                         &alpha,
                                                                         inputTensorDescriptor->descriptor(),
                                                                         inputTensor->begin(),
                                                                         outputTensorDescriptor->descriptor(),
                                                                         delta->begin(),
                                                                         convolutionDescriptor_,
                                                                         algo,
                                                                         d_workspace,
                                                                         workspace_size,
                                                                         &beta,
                                                                         filterDescriptor->descriptor(),
                                                                         dW->begin());

        if (bwd_filter_status == CUDNN_STATUS_NOT_SUPPORTED) {
            *result = {0, 0, BENCHMARK_NOT_SUPPORTED};
            bwd_filter_result.push_back({algo, result});
            return;
        } else if (bwd_filter_status != CUDNN_STATUS_SUCCESS) {
            *result = {0, 0, BENCHMARK_ERROR};
            bwd_filter_result.push_back({algo, result});
            return;
        }

    }
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    fwd_time = std::chrono::duration<double, std::micro>(end - start).count() / num_repeats;
    cudaFree(d_workspace);

    *result = {fwd_time, workspace_size, BENCHMARK_SUCCESS};
    bwd_filter_result.push_back({algo, result});
    return;
}

template<typename T>
void Benchmark<T>::backward_data(cudnnConvolutionBwdDataAlgo_t algo, uint32_t num_repeats) {
    assert(kernelTensor);
    assert(dX);
    assert(delta);


    size_t workspace_size;
    benchmarkResult *result = (benchmarkResult *) malloc(sizeof(benchmarkResult));
    try {
        workspace_size = bwd_data_workspace_size(algo);
    } catch (std::exception &exception) {
        *result = {0, 0, BENCHMARK_NOT_SUPPORTED};
        bwd_data_result.push_back({algo, result});
        return;
    }

    void *d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_size);

    double fwd_time = 0;
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        cudnnStatus_t bwd_data_status = cudnnConvolutionBackwardData(cudnn,
                                                                     &alpha,
                                                                     filterDescriptor->descriptor(),
                                                                     kernelTensor->begin(),
                                                                     outputTensorDescriptor->descriptor(),
                                                                     delta->begin(),
                                                                     convolutionDescriptor_,
                                                                     algo,
                                                                     d_workspace,
                                                                     workspace_size,
                                                                     &beta,
                                                                     inputTensorDescriptor->descriptor(),
                                                                     dX->begin());

        if (bwd_data_status == CUDNN_STATUS_NOT_SUPPORTED) {
            *result = {0, 0, BENCHMARK_NOT_SUPPORTED};
            bwd_data_result.push_back({algo, result});
            return;
        } else if (bwd_data_status != CUDNN_STATUS_SUCCESS) {
            *result = {0, 0, BENCHMARK_ERROR};
            bwd_data_result.push_back({algo, result});
            return;
        }
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    fwd_time = std::chrono::duration<double, std::micro>(end - start).count() / num_repeats;
    cudaFree(d_workspace);

    *result = {fwd_time, workspace_size, BENCHMARK_SUCCESS};
    bwd_data_result.push_back({algo, result});
    return;
}

template<typename T>
void Benchmark<T>::forward_algorythms(uint32_t num_repeats) {
    forward(CUDNN_CONVOLUTION_FWD_ALGO_GEMM, num_repeats);
    forward(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, num_repeats);
    forward(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, num_repeats);
    forward(CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, num_repeats);
    forward(CUDNN_CONVOLUTION_FWD_ALGO_FFT, num_repeats);
    forward(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING, num_repeats);
    forward(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, num_repeats);
    forward(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, num_repeats);
}

template<typename T>
void Benchmark<T>::backward_filter_algorythms(uint32_t num_repeats) {
    backward_filter(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, num_repeats);
    backward_filter(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, num_repeats);
    backward_filter(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3, num_repeats);
    backward_filter(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT, num_repeats);
    backward_filter(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING, num_repeats);
}

template<typename T>
void Benchmark<T>::backward_data_algorythms(uint32_t num_repeats) {
    backward_data(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, num_repeats);
    backward_data(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, num_repeats);
    backward_data(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, num_repeats);
    backward_data(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING, num_repeats);
    backward_data(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD, num_repeats);
    backward_data(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED, num_repeats);
}

template<typename T>
void Benchmark<T>::calculate_workspace_benchmark(uint32_t num_repeats) {
    assert(inputTensorDescriptor);
    assert(outputTensorDescriptor);
    assert(filterDescriptor);

    auto formatInputTensor = inputTensorDescriptor->format();
    auto formatOutputTensor = outputTensorDescriptor->format();
    auto formatFilter = filterDescriptor->format();

    inputTensor = new Tensor<T>(
            {formatInputTensor.N, formatInputTensor.H, formatInputTensor.W, formatInputTensor.C});
    outputTensor = new Tensor<T>(
            {formatOutputTensor.N, formatOutputTensor.H, formatOutputTensor.W, formatOutputTensor.C});
    kernelTensor = new Tensor<T>({formatFilter.N, formatFilter.H, formatFilter.W, formatFilter.C});

    delta = new Tensor<T>(
            {formatOutputTensor.N, formatOutputTensor.H, formatOutputTensor.W, formatOutputTensor.C});
    dW = new Tensor<T>({formatFilter.N, formatFilter.H, formatFilter.W, formatFilter.C});
    dX = new Tensor<T>({formatInputTensor.N, formatInputTensor.H, formatInputTensor.W, formatInputTensor.C});

    inputTensor->rand(curand_gen);
    kernelTensor->rand(curand_gen);
    delta->rand(curand_gen);

    forward_algorythms(num_repeats);
    backward_filter_algorythms(num_repeats);
    backward_data_algorythms(num_repeats);

    delete inputTensor;
    delete outputTensor;
    delete kernelTensor;
    delete delta;
    delete dW;
    delete dX;
}

template<typename T>
void Benchmark<T>::workspace_benchmark() {

}

template<typename T>
void Benchmark<T>::benchmark(benchmarkRow &benchmarkInput, uint32_t num_repeats) {
    fwd_result.clear();
    bwd_data_result.clear();
    bwd_filter_result.clear();
    this->benchmark_row = &benchmarkInput;

    cudnnDataType_t dataType;
    if (std::is_same<T, float>::value) {
        dataType = CUDNN_DATA_FLOAT;
    } else if (std::is_same<T, double>::value) {
        dataType = CUDNN_DATA_DOUBLE;
    } else if (std::is_same<T, uint16_t>::value) {
        dataType = CUDNN_DATA_HALF;
    } else if (std::is_same<T, int32_t>::value) {
        dataType = CUDNN_DATA_INT32;
    } else if (std::is_same<T, int8_t>::value) {
        dataType = CUDNN_DATA_INT8;
    } else if (std::is_same<T, uint8_t>::value) {
        dataType = CUDNN_DATA_UINT8;
    } else {
        throw new std::runtime_error("Cannot find supported format");
    }

    Format formatInputTensor = {
            benchmarkInput.n,
            benchmarkInput.c,
            benchmarkInput.h,
            benchmarkInput.w,
            benchmarkInput.inputTensorFormat
    };

    Format formatOutputTensor = {
            benchmarkInput.n,
            benchmarkInput.k,
            benchmarkInput.out_h,
            benchmarkInput.out_w,
            benchmarkInput.outputTensorFormat
    };

    Format formatFilter = {
            benchmarkInput.k,
            benchmarkInput.c,
            benchmarkInput.r,
            benchmarkInput.s,
            benchmarkInput.filterFormat
    };

    inputTensorDescriptor = new TensorDescriptor(formatInputTensor, dataType);
    outputTensorDescriptor = new TensorDescriptor(formatOutputTensor, dataType);
    filterDescriptor = new FilterDescriptor(formatFilter, dataType);


    CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&convolutionDescriptor_));
    CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(convolutionDescriptor_,
                                                      benchmarkInput.pad_h,
                                                      benchmarkInput.pad_w,
                                                      benchmarkInput.stride_h,
                                                      benchmarkInput.stride_w,
                                                      1,
                                                      1,
                                                      CUDNN_CONVOLUTION,
                                                      dataType));

    cudnnSetConvolutionMathType(convolutionDescriptor_, CUDNN_TENSOR_OP_MATH);

    switch (operation_mode) {
        case CALCULATION_AND_WORKSPACE_SIZE_MODE:
            calculate_workspace_benchmark(num_repeats);
            break;
        case ONLY_WORKSPACE_SIZE_MODE:
            workspace_benchmark();
            break;
    }

    delete inputTensorDescriptor;
    delete outputTensorDescriptor;
    delete filterDescriptor;

    CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(convolutionDescriptor_));
}

template<typename T>
void Benchmark<T>::run(std::string file_name, bool all_formats, benchmarkOperationMode operation_mode, uint32_t num_repeats,
                       cudnnTensorFormat_t input_format, cudnnTensorFormat_t output_format,
                       cudnnTensorFormat_t kernel_format) {

    auto benchmark_rows = parser::readInputDataFile(file_name);
    parser::openOutFile();

    Benchmark<T> benchmark(operation_mode);
    for (auto row : benchmark_rows) {
        if (!all_formats){
            row.inputTensorFormat = input_format;
            row.outputTensorFormat = output_format;
            row.filterFormat = kernel_format;

            try {
                benchmark.benchmark(row, num_repeats);
                parser::writeBenchmarkResult(benchmark);
            } catch (std::exception &e) {
                std::cerr << "Exception: " << e.what() << std::endl;
            }
        } else {
            row.inputTensorFormat = CUDNN_TENSOR_NCHW;
            row.outputTensorFormat = CUDNN_TENSOR_NCHW;
            row.filterFormat = CUDNN_TENSOR_NCHW;

            try {
                benchmark.benchmark(row, num_repeats);
                parser::writeBenchmarkResult(benchmark);
            } catch (std::exception &e) {
                std::cerr << "Exception: " << e.what() << std::endl;
            }

            row.inputTensorFormat = CUDNN_TENSOR_NHWC;
            row.outputTensorFormat = CUDNN_TENSOR_NHWC;
            row.filterFormat = CUDNN_TENSOR_NHWC;

            try {
                benchmark.benchmark(row, num_repeats);
                parser::writeBenchmarkResult(benchmark);
            } catch (std::exception &e) {
                std::cerr << "Exception: " << e.what() << std::endl;
            }

            row.inputTensorFormat = CUDNN_TENSOR_NCHW_VECT_C;
            row.outputTensorFormat = CUDNN_TENSOR_NCHW_VECT_C;
            row.filterFormat = CUDNN_TENSOR_NCHW_VECT_C;

            try {
                benchmark.benchmark(row, num_repeats);
                parser::writeBenchmarkResult(benchmark);
            } catch (std::exception &e) {
                std::cerr << "Exception: " << e.what() << "THIS FORMAT NOT SUPPORT CURRENT DATA TYPE" << std::endl;
            }
        }
    }
    parser::closeOutFile();
}

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "ERROR ARGS PROGRAM: \n"
                     "file_name - name of input file with convolution cases\n"
                     "data_type - type of data values (like fp16 and etc)\n"
                     "all_format - use all cudnn data format (true/false)\n"
                     "only_workspace - benchmark only workspace size\n"
                     "num_repeats - number of repeats per one algorithm\n"
                     "input_tensor_data_format - format of input tensor\n"
                     "output_tensor_data_format - format of output tensor\n"
                     "kernel_tensor_data_format - format of kernel tensor\n" << std::endl;
        return 1;

    }

    std::string file_name = argv[1];
    std::string data_type_name = argv[2];
    bool all_formats = static_cast<bool>(std::stoi(argv[3]));
    benchmarkOperationMode operation_mode = static_cast<benchmarkOperationMode>(std::stoi(argv[4]));
    uint32_t num_repeats = static_cast<uint32_t>(std::stoi(argv[5]));

    if ( !all_formats && (argc < 9) ) {
        std::cerr << "input_tensor_data_format - format of input tensor\n"
                     "output_tensor_data_format - format of output tensor\n"
                     "kernel_tensor_data_format - format of kernel tensor\n" << std::endl;
        return 1;
    }

    cudnnTensorFormat_t input_format;
    cudnnTensorFormat_t output_format;
    cudnnTensorFormat_t kernel_format;
    if (!all_formats){
        input_format = get_data_format_by_name(argv[6]);
        output_format = get_data_format_by_name(argv[7]);
        kernel_format = get_data_format_by_name(argv[8]);
    }

    if (data_type_name.compare("fp16") == 0)
        Benchmark<uint32_t>::run(file_name, all_formats, operation_mode, num_repeats, input_format, output_format, kernel_format);
    else if (data_type_name.compare("fp32") == 0)
        Benchmark<float>::run(file_name, all_formats, operation_mode, num_repeats, input_format, output_format, kernel_format);
    else if (data_type_name.compare("fp64") == 0)
        Benchmark<double>::run(file_name, all_formats, operation_mode, num_repeats, input_format, output_format, kernel_format);
    else throw new std::runtime_error("Data type not supported");

    return 0;
}