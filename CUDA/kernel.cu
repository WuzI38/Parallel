
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

#include "Headers\\methods_cuda.cuh"
#include "Headers\\algorithm_cuda.cuh"

#define MATRIX_SIZE 5 //4096
#define DAMPING_FACTOR 0.85
#define MAX_ITERATIONS 1024
#define PRECISION 15

#define NUM_TRIALS 10
#define FILENAMES {"matrix_1024.txt", "matrix_2048.txt", "matrix_4096.txt"}
#define FILESIZES {1024, 2048, 4096}
#define MATRIX_MAIN_FILE "matrix_5.txt"

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

int main()
{
    // Algorithm test part

    // Allocate memory on the device
    double* d_matrix;
    cudaMalloc(&d_matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));


    // Copy the matrix back to host memory
    double* matrix = new double[MATRIX_SIZE * MATRIX_SIZE];
    //cudaMemcpy(matrix, d_matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);


    // Load from file
    loadMatrixFromFile(matrix, MATRIX_SIZE, DIRECTORY, MATRIX_MAIN_FILE);
    //generateMatrix(matrix, MATRIX_SIZE, DAMPING_FACTOR);

    // Copy the matrix to device memory
    cudaMemcpy(d_matrix, matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Init vector pi(t)
    double* h_vector = new double[MATRIX_SIZE]; // local vector
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        h_vector[i] = 1.0 / MATRIX_SIZE;
    }

    // Allocate memory on the device for the vectors
    double* d_vector; // cuda vector
    double* d_newVector;
    cudaMalloc(&d_vector, MATRIX_SIZE * sizeof(double));
    cudaMalloc(&d_newVector, MATRIX_SIZE * sizeof(double));

    // Measure time
    auto start = std::chrono::high_resolution_clock::now();

    // Copy the initial vector to the device
    cudaMemcpy(d_vector, h_vector, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Algorithm's single iteration
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        multiplyVectorByMatrix(d_matrix, d_vector, d_newVector, MATRIX_SIZE); // matrix and vector multiplication 
        cudaDeviceSynchronize(); // Wait for GPU to finish
        if (compareVectors(d_vector, d_newVector, MATRIX_SIZE)) {
            break;
        }
        // Replace old vector with new vector
        cudaMemcpy(d_vector, d_newVector, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    // Copy the final vector back to host memory
    cudaMemcpy(h_vector, d_vector, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Algorithm execution time: " << elapsed.count() << "s\n";


    printMatrix(matrix, MATRIX_SIZE);
    std::cout << std::endl << std::endl;
    printVector(h_vector, MATRIX_SIZE);
    std::cout << std::endl << std::endl;

    // Check if probabilities sum to one
    double norm = 0;
    for (int x = 0; x < MATRIX_SIZE; x++) {
        norm += h_vector[x];
    }

    std::cout << norm << std::endl <<std::endl;

    delete[] matrix;
    delete[] h_vector;
    cudaFree(d_vector);
    cudaFree(d_newVector);
    cudaFree(d_matrix);

    measureMatrixGeneration(1024);
    measureMatrixGeneration(4096);
    measureMatrixGeneration(16384);

    // Measure algorithm execution time
    std::vector<std::string> filenames = FILENAMES;
    std::vector<int> filesizes = FILESIZES;

    for (std::size_t i = 0; i < filenames.size(); ++i) {
        measureAlgorithmExecution(filenames[i], filesizes[i]);
    }

    return 0;
}