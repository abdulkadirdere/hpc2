#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM = 32;
#define BLOCK_ROWS = 8;

int randomNumberGeneration(int upperBound, int lowerBound) {
    int num = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
    return num;
}

int **createData(int **array, int row, int column) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            array[i][j] = randomNumberGeneration(10, 1);
        }
    }
    return array;
}

int **createMatrix(int row, int column) {
    int **array = (int **)malloc(row * sizeof(int *));
    for (int i = 0; i < row; i++) {
        array[i] = (int *)malloc(column * sizeof(int *));
    }
    // create synthetic data for matrix
    array = createData(array, row, column);
    return array;
}

int **allocateMatrix(int row, int column) {
    int **array = (int **)malloc(row * sizeof(int *));
    for (int i = 0; i < row; i++) {
        array[i] = (int *)malloc(column * sizeof(int *));
    }
    return array;
}

void printArray(int **array, int row, int column) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            printf("%d ", array[i][j]);
        }
        printf("\n");
    }
}

__global__ void transpose_matrix(int **matrix){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("%d \n" , i);
}

int main(int argc, char** argv) {
    // matrix sizes 2^9, 2^10, 2^11, 2^12 = 512, 1024, 2048, 4096
    int matrix_dimensions[] = {8, 512, 1024, 2048, 4096};
    int matrix_dimension= matrix_dimensions[0];

    // create the matrix 
    int **h_matrix = createMatrix(matrix_dimension, matrix_dimension);
    int **d_matrix = allocateMatrix(matrix_dimension, matrix_dimension);

//  printArray(matrix, matrix_dimension, matrix_dimension);
    int size = matrix_dimension * sizeof(int);

    // allocate memory
    cudaMalloc((void **) &d_matrix, size);

    // copy memory from host to device
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    // size_t threads_per_block = 512;
    // size_t number_of_blocks = 32;
    // start the kernel
    transpose_matrix<<<8, matrix_dimension>>>(d_matrix);

    // copy back the results
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    // CUDA timing of event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // do some work on GPU

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time elpased: %3.6f ms\n", elapsedTime );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
}