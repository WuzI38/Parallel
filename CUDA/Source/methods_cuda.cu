#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <iomanip>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

// GPU - Init random number generator
__global__ void setupCurandStates(curandState* states, unsigned int seed) {
    // curandState must be different for each row/thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// GPU - matrix generation
__global__ void generateMatrixKernel(double* matrix, int n, double B, curandState* globalState) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    double dampingValue = (1.0 - B) / n;
    curandState localState = globalState[idx];
    // Each row is handled by a separate 
    double* row = &matrix[idx * n];

    // Initialize the row with the damping value
    for (int j = 0; j < n; ++j) {
        row[j] = dampingValue;
    }

    // Generate random edges
    int edges = 1 + curand(&localState) % (n - 1);
    double weight = B / edges;

    for (int j = 0; j < edges; ++j) {
        int index = curand(&localState) % n;
        row[index] += weight;
    }

    globalState[idx] = localState;
}

// CPU
void generateMatrix(double* matrix, int n, double B) {
    // Allocate memory for random number generator and initialize it
    curandState* devStates;
    cudaMalloc(&devStates, n * sizeof(curandState));

    // Generate seed
    unsigned int seed = time(NULL);
    setupCurandStates << <(n + 255) / 256, 256 >> > (devStates, seed);

    // Generate matrix on GPU
    // We use 256 threads per one block, as I've heard that is ok for NVIDIA GPUs
    generateMatrixKernel << <(n + 255) / 256, 256 >> > (matrix, n, B, devStates);

    cudaFree(devStates);
}

// This method is not a part of an algoritm, thus it is not implemented as a parallel method
void printMatrix(double* matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

// This method is not a part of an algoritm, thus it is not implemented as a parallel method
void printVector(double* vector, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

void saveMatrixToFile(double* matrix, int n, const std::string& directory, const std::string& filename, int precision = 15) {
    std::ofstream file(directory + filename);
    if (file.is_open()) {
        file << std::fixed << std::setprecision(precision);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file << matrix[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else {
        std::cout << "Cannot open the requested file" << std::endl;
    }
}

void loadMatrixFromFile(double* matrix, int n, const std::string& directory, const std::string& filename) {
    std::ifstream file(directory + filename);
    if (file.is_open()) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file >> matrix[i * n + j];
            }
        }
        file.close();
    }
    else {
        std::cout << "Cannot open the requested file" << std::endl;
    }
}
