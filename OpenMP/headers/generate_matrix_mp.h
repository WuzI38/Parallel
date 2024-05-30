#pragma once

#include <omp.h>
#include <ctime>
#include <vector>
#include <algorithm>

void generateMatrixMP(double** matrix, int n, double B);