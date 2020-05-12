#include <stdio.h>
#include <stdlib.h>

int WIDTH = 4;
int TILE_WIDTH = 2;

int randomNumberGeneration(int upperBound, int lowerBound) {
    int num = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
    return num;
}

int *createData(int *array, int num_element) {
    for (int i = 0; i < num_element; i++) {
        array[i] = randomNumberGeneration(9, 0);
    }
    return array;
}

int *createMatrix(int num_element) {
    int *array = (int *)malloc(num_element * sizeof(int *));
    // create synthetic data for matrix
    array = createData(array, num_element);
    return array;
}

int *allocateMatrix(int num_element) {
    int *array = (int *)malloc(num_element * sizeof(int *));
    return array;
}

void printArray(int *array, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", array[(i * width) + j]);
        }
        printf("\n");
    }
}

__global__ void global_transpose_matrix(int *o_data, const int *i_data, int tile_width){
    int x = blockIdx.x * blockDim.x + threadIdx.x; // blockDim = tile_width
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    for (int j = 0; j < blockDim.x; j+= width){
        o_data[x*width + (y+j)] = i_data[(y+j)*width + x];
    }
}

int main() {
    // matrix sizes 2^9, 2^10, 2^11, 2^12 = 512, 1024, 2048, 4096
    // int matrix_dimensions[] = {32, 512, 1024, 2048, 4096};
    // int matrix_dimension = matrix_dimensions[0];
    int num_elemet = WIDTH * WIDTH;

    // create the matrix 
    int *h_input_matrix = createMatrix(num_elemet);
    int *result_matrix = allocateMatrix(num_elemet);
    int *d_input_matrix;
    int *d_output_matrix;

    printArray(h_input_matrix, WIDTH);
    int memory_space_required = num_elemet * sizeof(int);

    // allocate memory
    cudaMalloc((void **) &d_input_matrix, memory_space_required);
    cudaMalloc((void **) &d_output_matrix, memory_space_required);

    // CUDA timing of event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // copy memory from host to device
    cudaMemcpy(d_input_matrix, h_input_matrix, memory_space_required, cudaMemcpyHostToDevice);

    // dimensions for the kernel
    dim3 Grid_Dim(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH);
    dim3 Block_Dim(TILE_WIDTH, TILE_WIDTH);

    // do some work on GPU
    // start the kernel
    cudaEventRecord(start);
    global_transpose_matrix<<<Grid_Dim, Block_Dim>>>(d_output_matrix, d_input_matrix, TILE_WIDTH);
    cudaEventRecord(stop);

    // copy back the results
    cudaMemcpy(result_matrix, d_output_matrix, memory_space_required, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time elpased: %3.6f ms \n", elapsedTime );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printArray(result_matrix, WIDTH);

    cudaFree(h_input_matrix);
    cudaFree(result_matrix);
    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix);
}