
# pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#include "..\\Headers\\methods_cuda.cuh"
#include "..\\Headers\\algorithm_cuda.cuh"

#define NUM_TRIALS 10
#define FILENAMES {"matrix_1024.txt", "matrix_2048.txt", "matrix_4096.txt"}
#define FILESIZES {1024, 2048, 4096}
#define MATRIXSIZES {1024, 1024, 4096}

#define DAMPING_FACTOR 0.85
#define MAX_ITERATIONS 1024
#define PRECISION 15

#define DIRECTORY "Data\\"

void measureMatrixGeneration(int matrixSize) {
    double totalDuration = 0.0;
    for (int i = 0; i < NUM_TRIALS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // Allocate memory on the device
        double* d_matrix;
        cudaMalloc(&d_matrix, matrixSize * matrixSize * sizeof(double));

        // Generate matrix
        generateMatrix(d_matrix, matrixSize, DAMPING_FACTOR);

        cudaFree(d_matrix);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        totalDuration += elapsed.count();
    }

    double averageDuration = totalDuration / NUM_TRIALS;
    std::cout << "Average matrix generation time for size " << matrixSize << ": " << averageDuration << "s\n";
}

void measureAlgorithmExecution(const std::string& filename, int matrixSize) {
    double totalDuration = 0.0;

    double* matrix = new double[matrixSize * matrixSize];
    loadMatrixFromFile(matrix, matrixSize, DIRECTORY, filename);

    for (int i = 0; i < NUM_TRIALS; ++i) {
        // Load from file

        double* d_matrix;
        double* d_vector; // cuda vector
        double* d_newVector;

        // Init vector pi(t)
        double* h_vector = new double[matrixSize]; // local vector
        for (int i = 0; i < matrixSize; ++i) {
            h_vector[i] = 1.0 / matrixSize;
        }

        auto start = std::chrono::high_resolution_clock::now();
        // Copy the matrix to device memory

        cudaMalloc(&d_matrix, matrixSize * matrixSize * sizeof(double));
        cudaMemcpy(d_matrix, matrix, matrixSize * matrixSize * sizeof(double), cudaMemcpyHostToDevice);

        // Allocate memory on the device for the vectors

        cudaMalloc(&d_vector, matrixSize * sizeof(double));
        cudaMalloc(&d_newVector, matrixSize * sizeof(double));

        // Copy the initial vector to the device
        cudaMemcpy(d_vector, h_vector, matrixSize * sizeof(double), cudaMemcpyHostToDevice);

        // Algorithm's single iteration
        for (int i = 0; i < MAX_ITERATIONS; ++i) {
            multiplyVectorByMatrix(d_matrix, d_vector, d_newVector, matrixSize); // matrix and vector multiplication 
            cudaDeviceSynchronize(); // Wait for GPU to finish
            if (compareVectors(d_vector, d_newVector, matrixSize)) {
                break;
            }
            // Replace old vector with new vector
            cudaMemcpy(d_vector, d_newVector, matrixSize * sizeof(double), cudaMemcpyDeviceToDevice);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        totalDuration += elapsed.count();

        delete[] h_vector;
        cudaFree(d_vector);
        cudaFree(d_newVector);
        cudaFree(d_matrix);
    }

    delete[] matrix;

    double averageDuration = totalDuration / NUM_TRIALS;
    std::cout << "Average algorithm execution time for " << filename << ": " << averageDuration << "s\n";
}

void measure() {
    std::vector<int> matrixsizes = MATRIXSIZES;

    for (std::size_t i = 0; i < matrixsizes.size(); ++i) {
        measureMatrixGeneration(matrixsizes[i]);
    }

    // Measure algorithm execution time
    std::vector<std::string> filenames = FILENAMES;
    std::vector<int> filesizes = FILESIZES;

    for (std::size_t i = 0; i < filenames.size(); ++i) {
        measureAlgorithmExecution(filenames[i], filesizes[i]);
    }
}