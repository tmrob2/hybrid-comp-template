#include <iostream>
#include "hello.h"

__global__ void sayHello() {
    printf("Hello world from the GPU!\n");
}

int cudaHello(mythread::Problem& problem) {
    printf("Hello world from the CPU!\n");
    uint index;
    double x, y;
    problem.getProblemData(index, x, y);
    std::cout << "This problem is -> index: " << index << " (x,y): " << "(" << x << "," << y << ")\n";

    sayHello<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}

int makeMatrix(){
    int   num_rows         = 5;
    int   num_cols         = 4;
    int   nnz              = 11;
    int   ld               = num_cols;
    int   dense_size       = ld * num_rows;
    int   h_csr_offsets[]  = { 0, 3, 4, 7, 9, 11 };
    int   h_csr_columns[]  = { 0, 2, 3, 1, 0, 2, 3, 1, 3, 1, 2 };
    float h_csr_values[]   = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                               7.0f, 8.0f, 9.0f, 10.0f, 11.0f };
    float h_dense[]        = { 0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f };
    float h_dense_result[] = { 1.0f,  0.0f,  2.0f,  3.0f,
                               0.0f,  4.0f,  0.0f,  0.0f,
                               5.0f,  0.0f,  6.0f,  7.0f,
                               0.0f,  8.0f,  0.0f,  9.0f,
                               0.0f, 10.0f, 11.0f,  0.0f };

    //--------------------------------------------------------------------------
    // Device memory management
    int   *d_csr_offsets, *d_csr_columns;
    float *d_csr_values,  *d_dense;
    CHECK_CUDA( cudaMalloc((void**) &d_csr_offsets,
                           (num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int))         )
    CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float))       )
    CHECK_CUDA( cudaMalloc((void**) &d_dense,       dense_size * sizeof(float)))

    CHECK_CUDA( cudaMemcpy(d_csr_offsets, h_csr_offsets,
                           (num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_csr_columns, h_csr_columns, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_csr_values, h_csr_values, nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_dense, h_dense, dense_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(d_csr_offsets) )
    CHECK_CUDA( cudaFree(d_csr_columns) )
    CHECK_CUDA( cudaFree(d_csr_values) )
    CHECK_CUDA( cudaFree(d_dense) )
    return EXIT_SUCCESS;
}