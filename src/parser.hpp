//
// Created by slimakanzer on 01.04.19.
//

#if !defined(BENCHMARK_PARSER_H)
#define BENCHMARK_PARSER_H

#include <fstream>
#include <vector>
#include <iostream>
#include <cudnn.h>
#include "benchmark.hpp"

namespace parser {
    std::vector<benchmarkRow> readInputDataFile(std::string file_name) {
        std::ifstream infile(file_name);

        std::vector<benchmarkRow> benchmark_rows;

        std::string line;
        std::string substr;
        size_t index_end;
        while (std::getline(infile, line)) {
            if (!((line.rfind("//", 0) == 0) || (line.rfind("\t", 0) == 0))) {
                benchmarkRow benchmark_row;

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.w = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.h = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.c = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.n = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.k = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.s = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.r = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.pad_w = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.pad_h = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.stride_w = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.stride_h = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.out_w = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.out_h = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.input_stride_w = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.input_stride_h = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.filter_stride_w = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                index_end = line.find('\t');
                substr = line.substr(0, index_end);
                benchmark_row.filter_stride_h = std::stoi(substr.c_str());
                line = line.substr(index_end + 1, line.length());

                benchmark_rows.push_back(benchmark_row);
            }
        }
        return benchmark_rows;
    }

    template<typename T>
    class Parser {
        Benchmark<T> *benchmark_;
        std::ofstream outfile_stream_;
        std::string out_file_name_;

        void openOutFile() {
            outfile_stream_.open(out_file_name_, std::ios::app);

            outfile_stream_ << "input_format"
                       "\toutput_format"
                       "\tfilter_format"
                       "\tW"
                       "\tH"
                       "\tC"
                       "\tN"
                       "\tK"
                       "\tS"
                       "\tR"
                       "\tpad_w"
                       "\tpad_h"
                       "\tstride_w"
                       "\tstride_h"
                       "\tout_w"
                       "\tout_h"
                       "\tinput_stride_w"
                       "\tinput_stride_h"
                       "\tfilter_stride_w"
                       "\tfilter_stride_h";

            switch (benchmark_->operation_mode) {
                case CALCULATION_AND_WORKSPACE_SIZE_MODE:
                    outfile_stream_ << "\tFWD_GEMM"
                               "\tFWD_GEMM WORKSPACE"
                               "\tFWD_IMPLICIT_GEMM"
                               "\tFWD_IMPLICIT_GEMM WORKSPACE"
                               "\tFWD_PRECOMP_GEMM"
                               "\tFWD_PRECOMP_GEMM WORKSPACE"
                               "\tFWD_DIRECT"
                               "\tFWD_DIRECT WROKSPACE"
                               "\tFWD_FFT"
                               "\tFWD_FFT WORKSPACE"
                               "\tFWD_FFT_TILING"
                               "\tFWD_FFT_TILING WORKSPACE"
                               "\tFWD_WINOGRAD"
                               "\tFWD_WINOGRAD WORKSPACE"
                               "\tFWD_WINOGRAD_NONFUSED"
                               "\tFWD_WINOGRAD_NONFUSED WORKSPACE"
                               "\tBWD_FILTER_ALGO_0"
                               "\tBWD_FILTER_ALGO_0 WORKPACE"
                               "\tBWD_FILTER_ALGO_1"
                               "\tBWD_FILTER_ALGO_1 WORKSPACE"
                               "\tBWD_FILTER_ALGO_3"
                               "\tBWD_FILTER_ALGO_3 WORKSPACE"
                               "\tBWD_FILTER_FFT"
                               "\tBWD_FILTER_FFT WORKSPACE"
                               "\tBWD FILTER FFT_TILING"
                               "\tBWD FILTER FFT_TILING WORKSPACE"
                               "\tBWD_DATA_ALGO_0"
                               "\tBWD_DATA_ALGO_0 WORKSPACE"
                               "\tBWD_DATA_ALGO_1"
                               "\tBWD_DATA_ALGO_1 WORKSPACE"
                               "\tBWD_DATA_FFT"
                               "\tBWD_DATA_FFT WORKSPACE"
                               "\tBWD_DATA_FFT_TILING"
                               "\tBWD_DATA_FFT_TILING WORKSPACE"
                               "\tBWD_DATA_WINOGRAD"
                               "\tBWD_DATA_WINOGRAD WORKSPACE"
                               "\tBWD_DATA_WINOGRAD_NONFUSED"
                               "\tBWD_DATA_WINOGRAD_NONFUSED WORKSPACE"
                               << std::endl;
                    break;
                case ONLY_WORKSPACE_SIZE_MODE:
                    outfile_stream_ << "\tFWD_GEMM"
                               "\tFWD_IMPLICIT_GEMM WORKSPACE"
                               "\tFWD_PRECOMP_GEMM WORKSPACE"
                               "\tFWD_DIRECT WORKSPACE"
                               "\tFWD_FFT WORKSPACE"
                               "\tFWD_FFT_TILING WORKSPACE"
                               "\tFWD_WINOGRAD WORKSPACE"
                               "\tFWD_WINOGRAD_NONFUSED WORKSPACE"
                               "\tBWD_FILTER_ALGO_0 WORKSPACE"
                               "\tBWD_FILTER_ALGO_1 WORKSPACE"
                               "\tBWD_FILTER_ALGO_3 WORKSPACE"
                               "\tBWD_FILTER_FFT WORKSPACE"
                               "\tBWD FILTER FFT_TILING WORKSPACE"
                               "\tBWD_DATA_ALGO_0 WORKSPACE"
                               "\tBWD_DATA_ALGO_1 WORKSPACE"
                               "\tBWD_DATA_FFT WORKSPACE"
                               "\tBWD_DATA_FFT_TILING WORKSPACE"
                               "\tBWD_DATA_WINOGRAD WORKSPACE"
                               "\tBWD_DATA_WINOGRAD_NONFUSED WORKSPACE"
                               << std::endl;
            }
        }

        void writeBenchmarkResultCalculateMode(benchmarkResult &result) {
            switch (result.status) {
                case BENCHMARK_SUCCESS:
                    outfile_stream_ << result.time << "\t" << result.workspace_size << "\t";
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile_stream_ << "n/a\t\t";
                    break;
                case BENCHMARK_ERROR:
                    outfile_stream_ << "-\t" << result.workspace_size <<"\t";
                    break;
            }
        }

        void writeBenchmarkResultWorkspaceMode(benchmarkResult &result) {
            switch (result.status) {
                case BENCHMARK_SUCCESS:
                    outfile_stream_  << result.workspace_size << "\t";
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile_stream_ << "n/a\t";
                    break;
                case BENCHMARK_ERROR:
                    outfile_stream_ << "-\t";
                    break;
            }
        }

        void writeBenchmarkCalculateMode() {
            auto row = benchmark_->benchmark_row;
            outfile_stream_ << get_data_format_name(row->inputTensorFormat) << "\t" << get_data_format_name(row->outputTensorFormat)
                    << "\t" << get_data_format_name(row->filterFormat) << "\t" << row->w << "\t" << row->h << "\t" << row->c
                    << "\t" << row->n << "\t" << row->k << "\t" << row->s << "\t" << row->r << "\t" << row->pad_w
                    << "\t" << row->pad_h << "\t" << row->stride_w << "\t" << row->stride_h
                    << "\t" << row->out_w << "\t" << row->out_h << "\t" << row->input_stride_w << "\t"
                    << row->input_stride_h
                    << "\t" << row->filter_stride_w << "\t" << row->filter_stride_h << "\t";

            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_FWD_ALGO_DIRECT);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_FWD_ALGO_FFT);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);

            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING);

            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_0);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD);
            writeBenchmarkResultCalculateMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);

            outfile_stream_ << std::endl;
        }

        void writeBenchmarkOnlyWorkspaceSizeMode() {
            auto row = benchmark_->benchmark_row;

            outfile_stream_ << get_data_format_name(row->inputTensorFormat) << "\t"
                    << get_data_format_name(row->outputTensorFormat)
                    << "\t" << get_data_format_name(row->filterFormat) << "\t" << row->w << "\t" << row->h << "\t" << row->c
                    << "\t" << row->n << "\t" << row->k << "\t" << row->s << "\t" << row->r << "\t" << row->pad_w
                    << "\t" << row->pad_h << "\t" << row->stride_w << "\t" << row->stride_h
                    << "\t" << row->out_w << "\t" << row->out_h << "\t" << row->input_stride_w << "\t"
                    << row->input_stride_h
                    << "\t" << row->filter_stride_w << "\t" << row->filter_stride_h << "\t";

            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_FWD_ALGO_DIRECT);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_FWD_ALGO_FFT);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);

            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING);

            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_0);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD);
            writeBenchmarkResultWorkspaceMode(row->CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);

            outfile_stream_ << std::endl;
        }

    public:
        Parser(Benchmark<T> *benchmark, std::string out_file_name = "benchmark_result.txt") {
            this->benchmark_ = benchmark;
            this->out_file_name_ = out_file_name;
            openOutFile();
        }

        ~Parser() {
            closeOutFile();
        }

        void closeOutFile() {
            outfile_stream_.close();
        }

        void writeBenchmarkResult() {
            switch (benchmark_->operation_mode) {
                case CALCULATION_AND_WORKSPACE_SIZE_MODE:
                    writeBenchmarkCalculateMode();
                    break;
                case ONLY_WORKSPACE_SIZE_MODE:
                    writeBenchmarkOnlyWorkspaceSizeMode();
                    break;
            }
        }
    };
}

#endif //BENCHMARK_PARSER_H
