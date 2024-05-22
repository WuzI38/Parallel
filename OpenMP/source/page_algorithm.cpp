#include <cmath>

// Calculate new pi vector
void multiplyVectorByMatrix(double* vector, double** matrix, double* result, int n) {
    for (int i = 0; i < n; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            result[i] += matrix[j][i] * vector[j];
        }
    }
}

// Chceck if current and previous vectors are the same
bool compareVectors(double* vector1, double* vector2, int n) {
    double epsilon = 1e-6;
    for (int i = 0; i < n; ++i) {
        if (std::abs(vector1[i] - vector2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}