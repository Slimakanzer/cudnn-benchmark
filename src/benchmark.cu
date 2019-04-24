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
benchmarkResult Benchmark<T>::forward(cudnnConvolutionFwdAlgo_t algo, uint32_t num_repeats) {
    assert(inputTensor);
    assert(outputTensor);
    assert(kernelTensor);

    size_t workspace_size;
    try {
        workspace_size = fwd_workspace_size(algo);
    } catch (std::exception &exception) {
        std::cerr << "WORKSPACE SIZE: " << get_fwd_algo_name(algo) << " " << exception.what();
        return {0, 0, BENCHMARK_NOT_SUPPORTED};
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

        if (fwd_status != CUDNN_STATUS_SUCCESS) {
            std::cerr << "CONVOLUTION: CUDNN failure: " << cudnnGetErrorString(fwd_status) << "algo: " << get_fwd_algo_name(algo)
                      << " file: " << __FILE__ << " line: " << __LINE__ << std::endl;
            return {0, workspace_size, BENCHMARK_ERROR};
        }
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    fwd_time = std::chrono::duration<double, std::micro>(end - start).count() / num_repeats;
    cudaFree(d_workspace);

    return {fwd_time, workspace_size, BENCHMARK_SUCCESS};
}

template<typename T>
benchmarkResult Benchmark<T>::backward_filter(cudnnConvolutionBwdFilterAlgo_t algo, uint32_t num_repeats) {
    assert(inputTensor);
    assert(dW);
    assert(delta);

    size_t workspace_size;
    try {
        workspace_size = bwd_filter_workspace_size(algo);
    } catch (std::exception &exception) {
        std::cerr << "WORKSPACE SIZE: " << get_bwd_filter_algo_name(algo) << " " << exception.what();
        return {0, 0, BENCHMARK_NOT_SUPPORTED};
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

        if (bwd_filter_status != CUDNN_STATUS_SUCCESS) {
            std::cerr << "CONVOLUTION: CUDNN failure: " << cudnnGetErrorString(bwd_filter_status) << "algo: "
                      << get_bwd_filter_algo_name(algo) << " file: " << __FILE__ << " line: " << __LINE__ << std::endl;
            return {0, workspace_size, BENCHMARK_ERROR};
        }

    }
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    fwd_time = std::chrono::duration<double, std::micro>(end - start).count() / num_repeats;
    cudaFree(d_workspace);

    return {fwd_time, workspace_size, BENCHMARK_SUCCESS};
}

template<typename T>
benchmarkResult Benchmark<T>::backward_data(cudnnConvolutionBwdDataAlgo_t algo, uint32_t num_repeats) {
    assert(kernelTensor);
    assert(dX);
    assert(delta);

    size_t workspace_size;
    try {
        workspace_size = bwd_data_workspace_size(algo);
    } catch (std::exception &exception) {
        std::cerr << "WORKSPACE SIZE: " << get_bwd_data_algo_name(algo) << " " << exception.what();
        return {0, 0, BENCHMARK_NOT_SUPPORTED};
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

        if (bwd_data_status != CUDNN_STATUS_SUCCESS) {
            std::cerr << "CONVOLUTION: CUDNN failure: " << cudnnGetErrorString(bwd_data_status) << " algo: "
                      << get_bwd_data_algo_name(algo) << " file: " << __FILE__ << " line: " << __LINE__ << std::endl;
            return {0, workspace_size, BENCHMARK_ERROR};
        }
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    fwd_time = std::chrono::duration<double, std::micro>(end - start).count() / num_repeats;
    cudaFree(d_workspace);

    return {fwd_time, workspace_size, BENCHMARK_SUCCESS};
}

template<typename T>
benchmarkResult Benchmark<T>::forward_workspace(cudnnConvolutionFwdAlgo_t algo) {
    size_t workspace_size;
    try {
        workspace_size = fwd_workspace_size(algo);
        return {0, workspace_size, BENCHMARK_SUCCESS};
    } catch (std::exception &exception) {
        return {0, 0, BENCHMARK_NOT_SUPPORTED};
    }
}

template<typename T>
benchmarkResult Benchmark<T>::backward_filter_workspace(cudnnConvolutionBwdFilterAlgo_t algo) {
    size_t workspace_size;
    try {
        workspace_size = bwd_filter_workspace_size(algo);
        return {0, workspace_size, BENCHMARK_SUCCESS};
    } catch (std::exception &exception) {
        return {0, 0, BENCHMARK_NOT_SUPPORTED};
    }
}

template<typename T>
benchmarkResult Benchmark<T>::backward_data_workspace(cudnnConvolutionBwdDataAlgo_t algo) {
    size_t workspace_size;
    try {
        workspace_size = bwd_data_workspace_size(algo);
        return {0, workspace_size, BENCHMARK_SUCCESS};
    } catch (std::exception &exception) {
        return {0, 0, BENCHMARK_NOT_SUPPORTED};
    }
}

template<typename T>
void Benchmark<T>::forward_algorythms(uint32_t num_repeats) {
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_GEMM = forward(CUDNN_CONVOLUTION_FWD_ALGO_GEMM, num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = forward(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                                                      num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = forward(
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = forward(CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_FFT = forward(CUDNN_CONVOLUTION_FWD_ALGO_FFT, num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = forward(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING, num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = forward(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = forward(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                                                                          num_repeats);
}

template<typename T>
void Benchmark<T>::backward_filter_algorythms(uint32_t num_repeats) {
    benchmark_row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = backward_filter(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                                                         num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = backward_filter(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                                                                         num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = backward_filter(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
                                                                         num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = backward_filter(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
                                                                           num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = backward_filter(
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING, num_repeats);
}

template<typename T>
void Benchmark<T>::backward_data_algorythms(uint32_t num_repeats) {
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = backward_data(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = backward_data(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = backward_data(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
                                                                       num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = backward_data(
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING, num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = backward_data(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
                                                                            num_repeats);
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = backward_data(
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED, num_repeats);
}

template<typename T>
void Benchmark<T>::forward_algorythms_workspace() {
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_GEMM = forward_workspace(CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = forward_workspace(
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = forward_workspace(
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = forward_workspace(CUDNN_CONVOLUTION_FWD_ALGO_DIRECT);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_FFT = forward_workspace(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = forward_workspace(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = forward_workspace(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);
    benchmark_row->CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = forward_workspace(
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
}

template<typename T>
void Benchmark<T>::backward_filter_algorythms_workspace() {
    benchmark_row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = backward_filter_workspace(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0);
    benchmark_row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = backward_filter_workspace(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
    benchmark_row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = backward_filter_workspace(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3);
    benchmark_row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = backward_filter_workspace(
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT);
    benchmark_row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = backward_filter_workspace(
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING);
}

template<typename T>
void Benchmark<T>::backward_data_algorythms_workspace() {
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = backward_data_workspace(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0);
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = backward_data_workspace(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = backward_data_workspace(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT);
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = backward_data_workspace(
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING);
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = backward_data_workspace(
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD);
    benchmark_row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = backward_data_workspace(
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);
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
    forward_algorythms_workspace();
    backward_filter_algorythms_workspace();
    backward_data_algorythms_workspace();
}

template<typename T>
void Benchmark<T>::benchmark(benchmarkRow &benchmarkInput, uint32_t num_repeats) {
    this->benchmark_row = &benchmarkInput;

    cudnnDataType_t dataType;
    if (std::is_same<T, DATA_FLOAT>::value) {
        dataType = CUDNN_DATA_FLOAT;
    } else if (std::is_same<T, DATA_DOUBLE>::value) {
        dataType = CUDNN_DATA_DOUBLE;
    } else if (std::is_same<T, DATA_HALF_FLOAT>::value) {
        dataType = CUDNN_DATA_HALF;
    } else if (std::is_same<T, DATA_INT32>::value) {
        dataType = CUDNN_DATA_INT32;
    } else if (std::is_same<T, DATA_INT8>::value) {
        dataType = CUDNN_DATA_INT8;
    } else if (std::is_same<T, DATA_UINT8>::value) {
        dataType = CUDNN_DATA_UINT8;
    } else if (std::is_same<T, DATA_INT8x4>::value) {
        dataType = CUDNN_DATA_INT8x4;
    } else if (std::is_same<T, DATA_INT8x32>::value) {
        dataType = CUDNN_DATA_INT8x32;
    } else if (std::is_same<T, DATA_UINT8x4>::value) {
        dataType = CUDNN_DATA_UINT8x4;
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
    int n, c, h, w;

    CHECK_CUDNN_ERROR(cudnnGetConvolution2dForwardOutputDim(
            convolutionDescriptor_,
            inputTensorDescriptor->descriptor(),
            filterDescriptor->descriptor(),
            &n,
            &c,
            &h,
            &w));

    std::cerr << "OUT VALUES: " << h <<" " << w << " " << c << " " << n << std::endl;

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
void
Benchmark<T>::run(std::string file_name, std::string output_file_name, bool all_formats,
                  benchmarkOperationMode operation_mode, uint32_t num_repeats,
                  cudnnTensorFormat_t input_format, cudnnTensorFormat_t output_format,
                  cudnnTensorFormat_t kernel_format) {

    auto benchmark_rows = parser::readInputDataFile(file_name);

    Benchmark<T> benchmark(operation_mode);
    parser::Parser<T> parser(&benchmark, output_file_name);
    for (auto row : benchmark_rows) {
        if (!all_formats) {
            row.inputTensorFormat = input_format;
            row.outputTensorFormat = output_format;
            row.filterFormat = kernel_format;

            try {
                benchmark.benchmark(row, num_repeats);
                parser.writeBenchmarkResult();
            } catch (std::exception &e) {
                std::cerr << "Exception: " << e.what() << std::endl;
            }
        } else {
            row.inputTensorFormat = CUDNN_TENSOR_NCHW;
            row.outputTensorFormat = CUDNN_TENSOR_NCHW;
            row.filterFormat = CUDNN_TENSOR_NCHW;

            try {
                benchmark.benchmark(row, num_repeats);
                parser.writeBenchmarkResult();
            } catch (std::exception &e) {
                std::cerr << "Exception: " << e.what() << std::endl;
            }

            row.inputTensorFormat = CUDNN_TENSOR_NHWC;
            row.outputTensorFormat = CUDNN_TENSOR_NHWC;
            row.filterFormat = CUDNN_TENSOR_NHWC;

            try {
                benchmark.benchmark(row, num_repeats);
                parser.writeBenchmarkResult();
            } catch (std::exception &e) {
                std::cerr << "Exception: " << e.what() << std::endl;
            }

            row.inputTensorFormat = CUDNN_TENSOR_NCHW_VECT_C;
            row.outputTensorFormat = CUDNN_TENSOR_NCHW_VECT_C;
            row.filterFormat = CUDNN_TENSOR_NCHW_VECT_C;

            try {
                benchmark.benchmark(row, num_repeats);
                parser.writeBenchmarkResult();
            } catch (std::exception &e) {
                std::cerr << "Exception: " << e.what() << "THIS FORMAT NOT SUPPORT CURRENT DATA TYPE" << std::endl;
            }
        }
    }
    parser.closeOutFile();
}

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "ERROR ARGS PROGRAM: \n"
                     "file_name - name of input file with convolution cases\n"
                     "file_name_output - name of output file with benchmark result\n"
                     "data_type - type of data values (like fp16 and etc)\n"
                     "all_format - use all cudnn data format (true/false)\n"
                     "operation_mode - benchmark operation mode\n"
                     "num_repeats - number of repeats per one algorithm\n"
                     "input_tensor_data_format - format of input tensor\n"
                     "output_tensor_data_format - format of output tensor\n"
                     "kernel_tensor_data_format - format of kernel tensor\n" << std::endl;
        return 1;

    }

    std::string file_name = argv[1];
    std::string output_file_name = argv[2];
    std::string data_type_name = argv[3];
    bool all_formats = static_cast<bool>(std::stoi(argv[4]));
    benchmarkOperationMode operation_mode = static_cast<benchmarkOperationMode>(std::stoi(argv[5]));
    uint32_t num_repeats = static_cast<uint32_t>(std::stoi(argv[6]));

    if (!all_formats && (argc < 10)) {
        std::cerr << "input_tensor_data_format - format of input tensor\n"
                     "output_tensor_data_format - format of output tensor\n"
                     "kernel_tensor_data_format - format of kernel tensor\n" << std::endl;
        return 1;
    }

    cudnnTensorFormat_t input_format;
    cudnnTensorFormat_t output_format;
    cudnnTensorFormat_t kernel_format;
    if (!all_formats) {
        input_format = get_data_format_by_name(argv[7]);
        output_format = get_data_format_by_name(argv[8]);
        kernel_format = get_data_format_by_name(argv[9]);
    }

    if (data_type_name.compare("fp16") == 0)
        Benchmark<DATA_HALF_FLOAT>::run(file_name, output_file_name, all_formats, operation_mode, num_repeats,
                                        input_format, output_format,
                                        kernel_format);
    else if (data_type_name.compare("fp32") == 0)
        Benchmark<DATA_FLOAT>::run(file_name, output_file_name, all_formats, operation_mode, num_repeats, input_format,
                                   output_format,
                                   kernel_format);
    else if (data_type_name.compare("fp64") == 0)
        Benchmark<DATA_DOUBLE>::run(file_name, output_file_name, all_formats, operation_mode, num_repeats, input_format,
                                    output_format,
                                    kernel_format);
    else if (data_type_name.compare("int8") == 0)
        Benchmark<DATA_INT8>::run(file_name, output_file_name, all_formats, operation_mode, num_repeats, input_format,
                                  output_format,
                                  kernel_format);
    else if (data_type_name.compare("uint8") == 0)
        Benchmark<DATA_UINT8>::run(file_name, output_file_name, all_formats, operation_mode, num_repeats, input_format,
                                   output_format,
                                   kernel_format);
    else if (data_type_name.compare("int32") == 0)
        Benchmark<DATA_INT32>::run(file_name, output_file_name, all_formats, operation_mode, num_repeats, input_format,
                                   output_format,
                                   kernel_format);
    else if (data_type_name.compare("int8x4") == 0)
        Benchmark<DATA_INT8x4>::run(file_name, output_file_name, all_formats, operation_mode, num_repeats, input_format,
                                    output_format,
                                    kernel_format);
    else if (data_type_name.compare("int8x32") == 0)
        Benchmark<DATA_INT8x32>::run(file_name, output_file_name, all_formats, operation_mode, num_repeats,
                                     input_format, output_format,
                                     kernel_format);
    else if (data_type_name.compare("uint8x4") == 0)
        Benchmark<DATA_UINT8x4>::run(file_name, output_file_name, all_formats, operation_mode, num_repeats,
                                     input_format, output_format,
                                     kernel_format);
    else std::cerr << "Data type not supported" << std::endl;

    return 0;
}