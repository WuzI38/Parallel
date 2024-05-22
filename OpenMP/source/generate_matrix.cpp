#include <ctime>
#include <vector>
#include <algorithm>

void generateMatrix(double** matrix, int n, double B) {
    std::srand(std::time(0));

    // Add damping factor 
    double dampingValue = (1.0 - B) / n;

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            matrix[i][j] = dampingValue;
        }
    }

    for (int i = 0; i < n; ++i) {
        std::vector<int> indices;
        // Randomly shuffle possible edge connections
        for (int j = 0; j < n; ++j) {
            if (i != j) indices.push_back(j); 
        }
        std::random_shuffle(indices.begin(), indices.end()); 
        // This generator assumes that form each page you can get to at least one page
        // This may not always be tha case, but even if it was the introduction of damping 
        // factor would take care of it
        int edges = 1 + std::rand() % (n - 1); 
        double weight = 1.0 / edges;
        for (int j = 0; j < edges; ++j) {
            matrix[i][indices[j]] = weight * B + dampingValue; 
        }
    }
}