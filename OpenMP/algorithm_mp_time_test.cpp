#include <iostream>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <cstdlib>
#include <chrono>

#include "headers\\methods_common.h"

#define MATRIX_SMALL_SIZE 1024
#define MATRIX_MEDIUM_SIZE 2048
#define MATRIX_BIG_SIZE 4096

#define MATRIX_SMALL_FILE "matrix_1024.txt"
#define MATRIX_MEDIUM_FILE "matrix_2048.txt" 
#define MATRIX_BIG_FILE "matrix_4096.txt"

#define ITERATIONS 10
#define MAX_ITERATIONS 1024

#define DIRECTORY "data\\"

int main(int argc, char* argv[])
{
    // Init matrix
    // I'm not making these sections parallel, as they may take longer to execute 
    // Because of their simplicity (I suspect this might be a false sharing type of problem)
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

    // Init vector pi(t)
    double* vectorSmall = new double[MATRIX_SMALL_SIZE]; 
    double* vectorMedium = new double[MATRIX_MEDIUM_SIZE]; 
    double* vectorBig = new double[MATRIX_BIG_SIZE]; 

    // Load matrices
    loadMatrixFromFile(matrixSmall, MATRIX_SMALL_SIZE, DIRECTORY, MATRIX_SMALL_FILE);
    loadMatrixFromFile(matrixMedium, MATRIX_MEDIUM_SIZE, DIRECTORY, MATRIX_MEDIUM_FILE);
    loadMatrixFromFile(matrixBig, MATRIX_BIG_SIZE, DIRECTORY, MATRIX_BIG_FILE);

    // Create vector pi(t+1)
    double* newVectorSmall = new double[MATRIX_SMALL_SIZE];
    double* newVectorMedium = new double[MATRIX_MEDIUM_SIZE];
    double* newVectorBig = new double[MATRIX_BIG_SIZE];

    // Algorithm
    iterateAndMeasure(vectorSmall, matrixSmall, newVectorSmall, MATRIX_SMALL_SIZE, ITERATIONS, MAX_ITERATIONS);
    iterateAndMeasure(vectorMedium, matrixMedium, newVectorMedium, MATRIX_MEDIUM_SIZE, ITERATIONS, MAX_ITERATIONS);
    iterateAndMeasure(vectorBig, matrixBig, newVectorBig, MATRIX_BIG_SIZE, ITERATIONS, MAX_ITERATIONS);

    // Delete data
    deleteMatrix(matrixSmall, MATRIX_SMALL_SIZE);
    deleteMatrix(matrixMedium, MATRIX_MEDIUM_SIZE);
    deleteMatrix(matrixBig, MATRIX_BIG_SIZE);

    delete[] vectorSmall;
    delete[] vectorMedium;
    delete[] vectorBig;

    delete[] newVectorSmall;
    delete[] newVectorMedium;
    delete[] newVectorBig;

    return 0;
}