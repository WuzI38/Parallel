#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <iomanip>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

__global__ void setupCurandStates(curandState* states, unsigned int seed);
__global__ void generateMatrixKernel(double* matrix, int n, double B, curandState* globalState);
void generateMatrix(double* matrix, int n, double B);
void printMatrix(double* matrix, int n);
void printVector(double* vector, int n);
void saveMatrixToFile(double* matrix, int n, const std::string& directory, const std::string& filename, int precision = 15);
void loadMatrixFromFile(double* matrix, int n, const std::string& directory, const std::string& filename);