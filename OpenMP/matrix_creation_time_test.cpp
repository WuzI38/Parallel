#include <iostream>
#include <cstdlib>

#include "headers\\methods_common.h"

#define MATRIX_SMALL_SIZE 1024
#define MATRIX_MEDIUM_SIZE 4096
#define MATRIX_BIG_SIZE 16384
#define DAMPING_FACTOR 0.85
#define ITERATIONS 10

int main(int argc, char* argv[])
{
    double** matrixSmall = new double*[MATRIX_SMALL_SIZE]; 
    for (int i = 0; i < MATRIX_SMALL_SIZE; ++i) {
        matrixSmall[i] = new double[MATRIX_SMALL_SIZE]; 
    }

    double** matrixMedium = new double*[MATRIX_MEDIUM_SIZE]; 
    for (int i = 0; i < MATRIX_MEDIUM_SIZE; ++i) {
        matrixMedium[i] = new double[MATRIX_MEDIUM_SIZE]; 
    }

    double** matrixBig = new double*[MATRIX_BIG_SIZE]; 
    for (int i = 0; i < MATRIX_BIG_SIZE; ++i) {
        matrixBig[i] = new double[MATRIX_BIG_SIZE]; 
    }

    // Generate and measure time for different matrix sizes
    generateAndMeasure(matrixSmall, MATRIX_SMALL_SIZE, ITERATIONS, DAMPING_FACTOR);
    generateAndMeasure(matrixMedium, MATRIX_MEDIUM_SIZE, ITERATIONS, DAMPING_FACTOR);
    generateAndMeasure(matrixBig, MATRIX_BIG_SIZE, ITERATIONS, DAMPING_FACTOR);
   
    // Delete each matrix
    deleteMatrix(matrixSmall, MATRIX_SMALL_SIZE);
    deleteMatrix(matrixMedium, MATRIX_MEDIUM_SIZE);
    deleteMatrix(matrixBig, MATRIX_BIG_SIZE);

    return 0;
}
