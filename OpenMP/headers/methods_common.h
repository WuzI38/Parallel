#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

#include "generate_matrix.h"
#include "generate_matrix_mpi.h"
#include "page_algorithm.h"
#include "page_algorithm_mpi.h"

void printMatrix(double** matrix, int n);
void printVector(double* vector, int n);
void saveMatrixToFile(double** matrix, int n, const std::string& directory, const std::string& filename, int precision=15);
void loadMatrixFromFile(double** matrix, int n, const std::string& directory, const std::string& filename);
void generateAndMeasure(double** matrix, int size, int iterations, double df);
void iterateAndMeasure(double* vector, double** matrix, double* newVector, int size, int iterations, int maxIterations);
void deleteMatrix(double** matrix, int size);