#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

#include "..\\headers\\generate_matrix.h"
#include "..\\headers\\generate_matrix_mpi.h"
#include "..\\headers\\page_algorithm.h"
#include "..\\headers\\page_algorithm_mpi.h"

// This method is not a part of an algoritm, thus it is not implemented as a parallel
void printMatrix(double** matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// This method is not a part of an algoritm, thus it is not implemented as a parallel
void printVector(double* vector, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

// Save and file methods should not be implemented as parallel, as input/output operation are sequential
void saveMatrixToFile(double** matrix, int n, const std::string& directory, const std::string& filename, int precision=15) {
    std::ofstream file(directory + filename);
    if (file.is_open()) {
        file << std::fixed << std::setprecision(precision);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file << matrix[i][j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Cannot open the requested file" << std::endl;
    }
}

void loadMatrixFromFile(double** matrix, int n, const std::string& directory, const std::string& filename) {
    std::ifstream file(directory + filename);
    if (file.is_open()) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                file >> matrix[i][j];
            }
        }
        file.close();
    } else {
        std::cout << "Cannot open the requested file" << std::endl;
    }
}

// Measure time and compare MPI and non MPI
void generateAndMeasure(double** matrix, int size, int iterations, double df) {
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    double timeSum = 0, timeSumMPI = 0;
    std::chrono::duration<double> elapsed;

    for(int x = 0; x < iterations; x++) {
        start = std::chrono::high_resolution_clock::now();
        generateMatrix(matrix, size, df);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        timeSum += elapsed.count();

        start = std::chrono::high_resolution_clock::now();
        generateMatrixMPI(matrix, size, df);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        timeSumMPI += elapsed.count();
    }

    std::cout << "Execution time average for size " << size << ": " << timeSum / iterations << "s\n";
    std::cout << "Execution time average for size " << size << " MPI: " << timeSumMPI / iterations << "s\n";
}

// Algorithm iterations
void iterateAndMeasure(double* vector, double** matrix, double* newVector, int size, int iterations, int maxIterations) {
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    double timeSum = 0, timeSumMPI = 0;
    std::chrono::duration<double> elapsed;

    // Reset vector
    for (int i = 0; i < size; ++i) {
        vector[i] = 1.0 / size;
    }

    // Perform algorithm without MPI
    for(int x = 0; x < iterations; x++) {
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < maxIterations; ++i) {
            multiplyVectorByMatrix(vector, matrix, newVector, size);
            if (compareVectors(vector, newVector, size)) {
                break;
            }
            std::copy(newVector, newVector+size, vector);
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        timeSum += elapsed.count();
    }

    // Reset vector
    for (int i = 0; i < size; ++i) {
        vector[i] = 1.0 / size;
    }

    // Perform algorithm wit MPI
    for(int x = 0; x < iterations; x++) {
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < maxIterations; ++i) {
            multiplyVectorByMatrixMPI(vector, matrix, newVector, size);
            if (compareVectorsMPI(vector, newVector, size)) {
                break;
            }
            std::copy(newVector, newVector+size, vector);
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        timeSumMPI += elapsed.count();
    }

    std::cout << "Execution time average for size " << size << ": " << timeSum / iterations << "s\n";
    std::cout << "Execution time average for size " << size << " MPI: " << timeSumMPI / iterations << "s\n";
}

// Delete matrix
void deleteMatrix(double** matrix, int size) {
    for (int i = 0; i < size; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}
