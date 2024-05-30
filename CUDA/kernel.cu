
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
// #include <vector>

#include "Headers\\methods_cuda.cuh"
#include "Headers\\algorithm_cuda.cuh"
#include "Headers\\time_test.cuh"

#define MATRIX_SIZE 5 //4096
#define DAMPING_FACTOR 0.85
#define MAX_ITERATIONS 1024
#define PRECISION 15

#define MATRIX_MAIN_FILE "matrix_5.txt"

#define DIRECTORY "Data\\"

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

    //measureMatrixGeneration(2048);
    //measureMatrixGeneration(8192);

    measure();

    return 0;
}