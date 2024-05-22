#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

// GPU - multiply vector by matrix, each thread is responsible for multiplying a single row by the vector
__global__ void multiplyVectorByMatrixKernel(double* matrix, double* vector, double* result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        sum += matrix[j * n + idx] * vector[j];
    }
    result[idx] = sum;
}

// Function to multiply a matrix by a vector
void multiplyVectorByMatrix(double* matrix, double* vector, double* result, int n) {
    multiplyVectorByMatrixKernel << <(n + 255) / 256, 256 >> > (matrix, vector, result, n);
}

// GPU - compare vectors
__global__ void compareVectorsKernel(double* vector1, double* vector2, bool* areEqual, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    double epsilon = 1e-6;
    if (abs(vector1[idx] - vector2[idx]) > epsilon) {
        *areEqual = false;
    }
}

// Function to compare two vectors
bool compareVectors(double* vector1, double* vector2, int n) {
    bool h_areEqual = true;
    bool* d_areEqual;
    cudaMalloc(&d_areEqual, sizeof(bool));
    cudaMemcpy(d_areEqual, &h_areEqual, sizeof(bool), cudaMemcpyHostToDevice);

    compareVectorsKernel << <(n + 255) / 256, 256 >> > (vector1, vector2, d_areEqual, n);

    cudaMemcpy(&h_areEqual, d_areEqual, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_areEqual);

    return h_areEqual;
}