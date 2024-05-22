#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void multiplyVectorByMatrixKernel(double* matrix, double* vector, double* result, int n);
__global__ void compareVectorsKernel(double* vector1, double* vector2, bool* areEqual, int n);
void multiplyVectorByMatrix(double* matrix, double* vector, double* result, int n);
bool compareVectors(double* vector1, double* vector2, int n);