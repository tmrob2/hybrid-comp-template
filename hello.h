//
// Created by thomas on 25/09/23.
//

#ifndef HYBRID_COMPUTING_HELLO_H
#define HYBRID_COMPUTING_HELLO_H

#include <iostream>
#include "library.h"

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse_v2.h>         // cusparseSparseToDense
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int cudaHello(mythread::Problem& problem);

int makeMatrix();

#endif //HYBRID_COMPUTING_HELLO_H
