#if !defined(CUDNN_HELPER_H)
#define CUDNN_HELPER_H

#include <cudnn.h>
#include <sstream>
#include <memory>
#include <vector>
#include "cudnn_data_type.h"

void throw_cudnn_err(cudnnStatus_t status, int line, const char *filename) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "CUDNN failure: " << cudnnGetErrorString(status) <<
           " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_CUDNN_ERROR(status) throw_cudnn_err(status, __LINE__, __FILE__)

std::string get_fwd_algo_name(cudnnConvolutionFwdAlgo_t algo) {
    switch (algo) {
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            return "FWD_IMPLICIT_GEMM           ";
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            return "FWD_PRECOMP_GEMM            ";
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return "FWD_GEMM                    ";
        case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            return "FWD_DIRECT                  ";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            return "FWD_FFT                     ";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            return "FWD_FFT_TILING              ";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            return "FWD_WINOGRAD                ";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            return "FWD_WINOGRAD_NONFUSED       ";
        default:
            return "FWD_NOT_FOUND_ALGO_NAME     ";
    }
}

std::string get_bwd_filter_algo_name(cudnnConvolutionBwdFilterAlgo_t algo) {
    switch (algo) {
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
            return "BWD_FILTER_ALGO_0           ";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
            return "BWD_FILTER_ALGO_1           ";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
            return "BWD_FILTER_ALGO_3           ";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
            return "BWD_FILTER_FFT              ";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
            return "BWD_FILTER_WINOGRAD_NONFUSED";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
            return "BWD FILTER FFT_TILING       ";
        default:
            return "BWD_FILTER_NOT_FOUND_ALGO_NAME";
    }
}

std::string get_bwd_data_algo_name(cudnnConvolutionBwdDataAlgo_t algo) {
    switch (algo) {
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
            return "BWD_DATA_ALGO_0             ";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
            return "BWD_DATA_ALGO_1             ";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
            return "BWD_DATA_FFT                ";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
            return "BWD_DATA_FFT_TILING         ";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
            return "BWD_DATA_WINOGRAD           ";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
            return "BWD_DATA_WINOGRAD_NONFUSED  ";
        default:
            return "BWD_DATA_NOT_FOUND_ALGO_NAME";
    }
}

std::string get_data_format_name(cudnnTensorFormat_t format) {
    switch (format) {
        case CUDNN_TENSOR_NCHW:
            return "NCHW";
        case CUDNN_TENSOR_NHWC:
            return "NHWC";
        case CUDNN_TENSOR_NCHW_VECT_C:
            return "NCHW_VECT_C";
        default:
            return "FORMAT_NOT_FOUND_NAME";
    }
}

cudnnTensorFormat_t get_data_format_by_name(std::string name) {
    if (name.compare("NCHW") == 0) return CUDNN_TENSOR_NCHW;
    else if (name.compare("NHWC") == 0) return CUDNN_TENSOR_NHWC;
    else if (name.compare("NCHW_VECT_C") == 0) return CUDNN_TENSOR_NCHW_VECT_C;
    else return CUDNN_TENSOR_NCHW;
}

struct Format {
    int N, C, H, W;
    cudnnTensorFormat_t format;
};

class TensorDescriptor {
    std::shared_ptr<cudnnTensorDescriptor_t> descriptor_;
    Format format_;
    cudnnDataType_t dataType_;

    struct TensorDescriptorDeleter {
        void operator()(cudnnTensorDescriptor_t *desc) {
            CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(*desc));
            delete desc;
        }
    };

public:

    TensorDescriptor(const Format &format, cudnnDataType_t dataType) {
        cudnnTensorDescriptor_t *desc = new cudnnTensorDescriptor_t;
        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(desc));
        CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(*desc,
                                                     format.format,
                                                     dataType,
                                                     format.N,
                                                     format.C,
                                                     format.H,
                                                     format.W));

        std::cerr << "H: " << format.H << "  W: " << format.W << "  C: " << format.C << "  N: " << format.N
                  << std::endl;
        descriptor_.reset(desc, TensorDescriptorDeleter());
        format_ = format;
        dataType_ = dataType;
    }

    cudnnTensorDescriptor_t descriptor() { return *descriptor_; };

    Format format() { return format_; };

    cudnnDataType_t dataType() { return dataType_; };
};

class FilterDescriptor {
    std::shared_ptr<cudnnFilterDescriptor_t> descriptor_;
    Format format_;
    cudnnDataType_t dataType_;

    struct FilterDescriptorDeleter {
        void operator()(cudnnFilterDescriptor_t *desc) {
            CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(*desc));
            delete desc;
        }
    };

public:
    FilterDescriptor(const Format &format, cudnnDataType_t dataType) {
        auto *desc = new cudnnFilterDescriptor_t;
        CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(desc));
        CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(*desc,
                                                     dataType,
                                                     format.format,
                                                     format.N,
                                                     format.C,
                                                     format.H,
                                                     format.W));
        std::cerr << "H: " << format.H << "  W: " << format.W << " C: " << format.C << "  K: " << format.N << std::endl;
        descriptor_.reset(desc, FilterDescriptorDeleter());
        format_ = format;
        dataType_ = dataType;
    }

    cudnnFilterDescriptor_t descriptor() { return *descriptor_; };

    Format format() { return format_; };

    cudnnDataType_t dataType() { return dataType_; };
};

#endif //CUDNN_HELPER_H