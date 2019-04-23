//
// Created by slimakanzer on 29.03.19.
//

#if !defined(BENCHMARK_TENSOR_H)
#define BENCHMARK_TENSOR_H

#include <vector>
#include <memory>
#include <numeric>
#include <curand.h>

template <typename T>
class Tensor {
    std::vector<int> dims_;
    int size_;

    T* ptr_;

public:
    Tensor(std::vector<int> dims) : dims_(dims) {
        size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
        cudaMalloc(&ptr_, sizeof(T) * size_);
    }

    ~Tensor() {
        cudaFree(ptr_);
    }

    T* begin() const { return ptr_; }
    T* end()   const { return ptr_ + size_; }
    int size() const { return size_; }
    std::vector<int> dims() const { return dims_; }

    // TODO need template variable for curand
    void rand(curandGenerator_t curand_gen) {

    }
};


#endif //BENCHMARK_TENSOR_H
