#pragma once

#include <omp.h>
#include <ctime>
#include <vector>
#include <algorithm>

void generateMatrixMPI(double** matrix, int n, double B);