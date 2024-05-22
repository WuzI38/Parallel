#include <cmath>

// Calculate new pi vector
void multiplyVectorByMatrixMPI(double* vector, double** matrix, double* result, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            result[i] += matrix[j][i] * vector[j];
        }
    }
}

// Chceck if current and previous vectors are the same
bool compareVectorsMPI2(double* vector1, double* vector2, int n) {
    double epsilon = 1e-6;
    bool areEqual = true;
    #pragma omp parallel for shared(areEqual)
    for (int i = 0; i < n; ++i) {
        if (!areEqual) {
            continue;
        }
        if (std::abs(vector1[i] - vector2[i]) > epsilon) {
            #pragma omp atomic write
            areEqual = false;
        }
    }
    return areEqual;
}

// Alternative version using reduction, might be a bit faster
bool compareVectorsMPI(double* vector1, double* vector2, int n) {
    double epsilon = 1e-6;
    bool areEqual = true;
    #pragma omp parallel for reduction(&:areEqual)
    for (int i = 0; i < n; ++i) {
        if (std::abs(vector1[i] - vector2[i]) > epsilon) {
            areEqual = false;
        }
    }
    return areEqual;
}