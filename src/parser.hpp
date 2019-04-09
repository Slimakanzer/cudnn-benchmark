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

#define OUT_FILE_NAME "benchmark_result.txt"

namespace parser {
    std::ofstream outfile;

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
                << "// input_format output_format filter_format W H C N K S R pad_w pad_h stride_w stride_h out_w out_h input_stride_w input_stride_h filter_stride_w filter_stride_h"
                << std::endl;
        outfile << "// ALGO STATUS TIME WORKSPACE" << std::endl;
    }

    void closeOutFile() {
        outfile.close();
    }

    template<typename T>
    void writeBenchmarkResult(Benchmark<T> &benchmark) {
        outfile << std::endl;
        auto row = benchmark.benchmark_row;
        outfile << get_data_format_name(row->inputTensorFormat) << " " << get_data_format_name(row->outputTensorFormat)
                << " " << get_data_format_name(row->filterFormat) << " " << row->w << " " << row->h << " " << row->c
                << " " << row->n << " " << row->k << " " << row->s << " " << row->r << " " << row->pad_w
                << " " << row->pad_h << " " << row->stride_w << " " << row->stride_h
                << " " << row->out_w << " " << row->out_h << " " << row->input_stride_w << " " << row->input_stride_h
                << " " << row->filter_stride_w << " " << row->filter_stride_h << std::endl;

        for (auto fwd_result : benchmark.fwd_result) {
            outfile << get_fwd_algo_name(fwd_result.algo) << " ";

            switch (fwd_result.result->status) {
                case BENCHMARK_SUCCESS:
                    outfile << "success " << fwd_result.result->time << " " << fwd_result.result->workspace_size
                            << std::endl;
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile << "n/a" << std::endl;
                    break;
                case BENCHMARK_ERROR:
                    outfile << "error" << std::endl;
                    break;
            }
        }

        for (auto bwd_filter : benchmark.bwd_filter_result) {
            outfile << get_bwd_filter_algo_name(bwd_filter.algo) << " ";

            switch (bwd_filter.result->status) {
                case BENCHMARK_SUCCESS:
                    outfile << "success " << bwd_filter.result->time << " " << bwd_filter.result->workspace_size
                            << std::endl;
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile << "n/a" << std::endl;
                    break;
                case BENCHMARK_ERROR:
                    outfile << "error" << std::endl;
                    break;
            }
        }

        for (auto bwd_data : benchmark.bwd_data_result) {
            outfile << get_bwd_data_algo_name(bwd_data.algo) << " ";

            switch (bwd_data.result->status) {
                case BENCHMARK_SUCCESS:
                    outfile << "success " << bwd_data.result->time << " " << bwd_data.result->workspace_size
                            << std::endl;
                    break;
                case BENCHMARK_NOT_SUPPORTED:
                    outfile << "n/a" << std::endl;
                    break;
                case BENCHMARK_ERROR:
                    outfile << "error" << std::endl;
                    break;
            }
        }

        outfile << std::endl;
    }
}

#endif //BENCHMARK_PARSER_H
