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

void measureMatrixGeneration(int matrixSize);
void measureAlgorithmExecution(const std::string& filename, int matrixSize);
void measure();