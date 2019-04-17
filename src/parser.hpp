//
// Created by slimakanzer on 01.04.19.
//

#ifndef BENCHMARK_PARSER_H
#define BENCHMARK_PARSER_H

#include <fstream>
#include <vector>
#include <iostream>
#include <cudnn.h>
#include "benchmark.hpp"

namespace parser {
    std::ofstream outfile;
    std::string OUT_FILE_NAME = "benchmark_result.txt";

    std::vector<benchmarkRow> readInputDataFile(std::string file_name) {
        std::ifstream infile(file_name);

        std::vector<benchmarkRow> benchmark_rows;

        std::string line;
        std::string substr;
        unsigned long index_end;
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

    void openOutFile() {
        outfile.open(OUT_FILE_NAME, std::ios::app);
        outfile
                << "\tinput_format\toutput_format\tfilter_format\tW\tH\tC\tN\tK\tS\tR\tpad_w\tpad_h\tstride_w\tstride_h\tout_w\tout_h\tinput_stride_w\tinput_stride_h\tfilter_stride_w\tfilter_stride_h"
                << std::endl;
    }

    void closeOutFile() {
        outfile.close();
    }

    template<typename T>
    void writeBenchmarkCalculateMode(Benchmark<T> &benchmark) {
        outfile << std::endl;
        auto row = benchmark.benchmark_row;
        outfile << "\t" << get_data_format_name(row->inputTensorFormat) << "\t" << get_data_format_name(row->outputTensorFormat)
                << "\t" << get_data_format_name(row->filterFormat) << "\t" << row->w << "\t" << row->h << "\t" << row->c
                << "\t" << row->n << "\t" << row->k << "\t" << row->s << "\t" << row->r << "\t" << row->pad_w
                << "\t" << row->pad_h << "\t" << row->stride_w << "\t" << row->stride_h
                << "\t" << row->out_w << "\t" << row->out_h << "\t" << row->input_stride_w << "\t"
                << row->input_stride_h
                << "\t" << row->filter_stride_w << "\t" << row->filter_stride_h << std::endl;

        for (auto fwd_result : benchmark.fwd_result) {
            outfile << get_fwd_algo_name(fwd_result.algo) << "\t";

            switch (fwd_result.result->status) {
                case BENCHMARK_SUCCESS:
                    outfile << fwd_result.result->time << "\t" << fwd_result.result->workspace_size
                            << std::endl;
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile << "n/a" << std::endl;
                    break;
                case BENCHMARK_ERROR:
                    outfile << "-" << std::endl;
                    break;
            }
        }

        for (auto bwd_filter : benchmark.bwd_filter_result) {
            outfile << get_bwd_filter_algo_name(bwd_filter.algo) << "\t";

            switch (bwd_filter.result->status) {
                case BENCHMARK_SUCCESS:
                    outfile << bwd_filter.result->time << "\t" << bwd_filter.result->workspace_size
                            << std::endl;
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile << "n/a" << std::endl;
                    break;
                case BENCHMARK_ERROR:
                    outfile << "-" << std::endl;
                    break;
            }
        }

        for (auto bwd_data : benchmark.bwd_data_result) {
            outfile << get_bwd_data_algo_name(bwd_data.algo) << "\t";

            switch (bwd_data.result->status) {
                case BENCHMARK_SUCCESS:
                    outfile << bwd_data.result->time << "\t" << bwd_data.result->workspace_size
                            << std::endl;
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile << "n/a" << std::endl;
                    break;
                case BENCHMARK_ERROR:
                    outfile << "-" << std::endl;
                    break;
            }
        }

        outfile << std::endl;
    }

    template<typename T>
    void writeBenchmarkOnlyWorkspaceSizeMode(Benchmark<T> &benchmark) {
        outfile << std::endl;
        auto row = benchmark.benchmark_row;

        outfile << "\t" << get_data_format_name(row->inputTensorFormat) << "\t" << get_data_format_name(row->outputTensorFormat)
                << "\t" << get_data_format_name(row->filterFormat) << "\t" << row->w << "\t" << row->h << "\t" << row->c
                << "\t" << row->n << "\t" << row->k << "\t" << row->s << "\t" << row->r << "\t" << row->pad_w
                << "\t" << row->pad_h << "\t" << row->stride_w << "\t" << row->stride_h
                << "\t" << row->out_w << "\t" << row->out_h << "\t" << row->input_stride_w << "\t"
                << row->input_stride_h
                << "\t" << row->filter_stride_w << "\t" << row->filter_stride_h << std::endl;

        for (auto fwd_result : benchmark.fwd_result) {
            outfile << get_fwd_algo_name(fwd_result.algo) << "\t";

            switch (fwd_result.result->status) {
                case BENCHMARK_SUCCESS:
                    outfile << fwd_result.result->workspace_size
                            << std::endl;
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile << "n/a" << std::endl;
                    break;
                case BENCHMARK_ERROR:
                    outfile << "-" << std::endl;
                    break;
            }
        }

        for (auto bwd_filter : benchmark.bwd_filter_result) {
            outfile << get_bwd_filter_algo_name(bwd_filter.algo) << "\t";

            switch (bwd_filter.result->status) {
                case BENCHMARK_SUCCESS:
                    outfile << bwd_filter.result->workspace_size
                            << std::endl;
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile << "n/a" << std::endl;
                    break;
                case BENCHMARK_ERROR:
                    outfile << "-" << std::endl;
                    break;
            }
        }

        for (auto bwd_data : benchmark.bwd_data_result) {
            outfile << get_bwd_data_algo_name(bwd_data.algo) << "\t";

            switch (bwd_data.result->status) {
                case BENCHMARK_SUCCESS:
                    outfile << bwd_data.result->workspace_size
                            << std::endl;
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile << "n/a" << std::endl;
                    break;
                case BENCHMARK_ERROR:
                    outfile << "-" << std::endl;
                    break;
            }
        }

        outfile << std::endl;
    }

    template<typename T>
    void writeBenchmarkResult(Benchmark<T> &benchmark) {
        switch (benchmark.operation_mode) {
            case CALCULATION_AND_WORKSPACE_SIZE_MODE:
                writeBenchmarkCalculateMode(benchmark);
                break;
            case ONLY_WORKSPACE_SIZE_MODE:
                writeBenchmarkOnlyWorkspaceSizeMode(benchmark);
                break;
        }
    }
}

#endif //BENCHMARK_PARSER_H
