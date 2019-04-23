//
// Created by slimakanzer on 20.04.19.
//

#if !defined(CUDNN_DATA_TYPE_H)
#define CUDNN_DATA_TYPE_H

typedef uint16_t DATA_HALF_FLOAT;
typedef float DATA_FLOAT;
typedef double DATA_DOUBLE;
typedef int8_t DATA_INT8;
typedef uint8_t DATA_UINT8;
typedef int32_t DATA_INT32;
typedef struct {
    int8_t a, b, c, d;
} DATA_INT8x4;
typedef struct {
    int64_t a,b,c,d;
} DATA_INT8x32;
typedef struct {
    uint8_t a, b, c, d;
} DATA_UINT8x4;

#endif //CUDNN_DATA_TYPE_H
