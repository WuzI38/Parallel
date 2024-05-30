#include "headers\\methods_common.h"

#define MATRIX_SIZE 4096
#define DAMPING_FACTOR 0.85
#define MAX_ITERATIONS 1024
#define PRECISION 15
#define DIRECTORY "data\\"

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

    // If 2 arguments were given save the matrix to a taxt file with given name
    if(argc == 2) {
        generateMatrixMP(matrix, MATRIX_SIZE, DAMPING_FACTOR);
        saveMatrixToFile(matrix, MATRIX_SIZE, DIRECTORY, argv[1], PRECISION);
        return 0;
    }
    else {
        return 0;
    }
    
    deleteMatrix(matrix, MATRIX_SIZE);

    return 0;
}