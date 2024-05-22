#include <iostream>
#include <algorithm>
#include <iterator>
#include <cstdlib>

#include "headers\\methods_common.h"

#define MATRIX_SIZE 5
#define MATRIX_FILE_NAME "matrix_5.txt"
#define DAMPING_FACTOR 0.85
#define MAX_ITERATIONS 1024
#define PRECISION 15

#define DIRECTORY "data\\" // ../Data/"

int main(int argc, char* argv[])
{
    // Init matrix
    // I'm not making these sections parallel, as they may take longer to execute 
    // Because of their simplicity (I suspect this might be a false sharing type of problem)
    double** matrix = new double*[MATRIX_SIZE]; 
    //#pragma omp parallel for
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        matrix[i] = new double[MATRIX_SIZE]; 
    }

    // Init vector pi(t)
    double* vector = new double[MATRIX_SIZE]; 
    //#pragma omp parallel for
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        vector[i] = 1.0 / MATRIX_SIZE;
    }

    // If 2 arguments were given just generate a new matrix and return matrix creation time
    loadMatrixFromFile(matrix, MATRIX_SIZE, DIRECTORY, MATRIX_FILE_NAME);

    // Create vector pi(t+1)
    double* newVector = new double[MATRIX_SIZE];

    // Algorithm's single iteration
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        multiplyVectorByMatrixMPI(vector, matrix, newVector, MATRIX_SIZE); // matrix and vector multiplication 
        if (compareVectorsMPI(vector, newVector, MATRIX_SIZE)) {
            break;
        }
        // Replace old vector with new vector
        std::copy(newVector, newVector+MATRIX_SIZE, vector);
    }

    printMatrix(matrix, MATRIX_SIZE);
    std::cout << std::endl << std::endl;
    printVector(vector, MATRIX_SIZE);
    std::cout << std::endl << std::endl;

    // Check if the probability distribution is normalized
    // This part is not parallel, this is also not a part of an algorithm
    double norm = 0;
    for(int x=0; x < MATRIX_SIZE; x++) {
        norm += newVector[x];
    }

    // Check if probabilities sum to one
    std::cout << norm << std::endl;
    
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] vector;
    delete[] newVector;

    return 0;
}