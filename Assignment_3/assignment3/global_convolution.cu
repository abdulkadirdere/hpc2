#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "helper/inc/helper_functions.h" // includes cuda.h and cuda_runtime_api.h
#include "helper/inc/helper_cuda.h" // helper functions for CUDA error check

const int size=512;
const int mask_size = 3;
const int offset = floor(mask_size/2);
const int padded_size = size + 2*offset;

const int TILE_WIDTH = 16;


__const__ double mask[mask_size][mask_size] = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
};

// gaussian
// __const__ double mask[mask_size][mask_size] = {
//     {0, 0, -1, 0, 0},
//     {0, -1, -2, -1, 0},
//     {-1, -2, 16, -2, -1},
//     {0, -1, -2, -1, 0},
//     {0, 0, -1, 0, 0},
// };

// edge detection
// __const__ double mask[mask_size][mask_size] = {
//     {-1, 0, 1},
//     {-2, 0, 2},
//     {-1, 0, 1},
// };

// sharpenning
// __const__ double mask[mask_size][mask_size] = {
//     {-1, -1, -1},
//     {-1,  9, -1},
//     {-1, -1, -1},
// };

void printArray(double **array, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%3.6f ", array[i][j]);
        }
        printf("\n");
    }
}

// void printArray(float *array, int r, int c) {
//     for (int i = 0; i < r; i++) {
//         for (int j = 0; j < c; j++) {
//             printf("%3.6f ", array[(i*r) +j]);
//         }
//         printf("\n");
//     }
// }

double **allocateMatrix(int m, int n) {
    double **array = (double **)malloc(m * sizeof(double *));
    for (int i=0; i<m; i++){
        array[i] = (double *)malloc(n * sizeof(double)); 
    }

    int zero = 0; 
    for (int i = 0; i <  m; i++){
        for (int j = 0; j < n; j++) {
            array[i][j] = zero;
        }
    }

    return array;
}

double **convert2D(float *input, unsigned int width, unsigned int height) {
    double **array = (double **)malloc(width * sizeof(double *));
    for (int i=0; i<width; i++){
        array[i] = (double *)malloc(height * sizeof(double)); 
    }

    int value = 0; 
    for (int i = 0; i <  width; i++){
        for (int j = 0; j < height; j++) {
            array[i][j] = input[value];
            value++;
        }
    }
    return array;
}

float *convert1D(double **input, unsigned int width, unsigned int height) {
    unsigned int size = width * height * sizeof(float);
    float *array = (float *)malloc(size * sizeof(float));

    int value = 0; 
    for (int i = 0; i <  width; i++){
        for (int j = 0; j < height; j++) {
            array[value] = (float)input[i][j];
            value++;
        }
    }
    return array;
}

double **padArray(double **input, double **output) {
    int range = padded_size - offset;
    // printf("%d \n", range);

    // pad the array
    for (int i = offset; i < range; i++) {
        for (int j = offset; j < range; j++) {
            output[i][j] = input[i-offset][j-offset];
        }
    }
    return output;
}

double **unpad(double **input, double **output) {
    int range = padded_size - offset;

    // unpad the array
    for (int i = 0; i < range; i++) {
        for (int j = 0; j < range; j++) {
            output[i][j] = input[i+offset][j+offset];
        }
    }
    return output;
}

double applyMask(double **array, int row, int col){
    int n_size = offset * 2 + 1;

    // neighbours of given location
    double **neighbours = allocateMatrix(n_size, n_size);

    // dynamically get the neighbours
    int n1 = 0;
    for (int r=row - 1; r <= row + offset; r++){
        int n2 = 0;
        for (int c =col - 1; c <= col + offset; c++){
            neighbours[n1][n2] = array[r][c];
            n2++;
        }
        n1++;
    }

    // element-wise matrix multiplication
    double **convolution = allocateMatrix(n_size, n_size);
    double value = 0;

    for (int r=0; r<3; r++){
        for(int c=0; c<3; c++){
            // printf("value: %3.6f \n", mask[1][1]);
            convolution[r][c] = mask[r][c] * neighbours[r][c];
            value = value + convolution[r][c];
        }
    }
    // printf("value: %3.6f \n", value);
    // printArray(convolution, offset, offset);

    return value;
}

double **serial_convolution(double **input, double **output){
    int range = padded_size - offset;
    // printf("range: %d \n", range);

    for (int i = offset; i<range; i++){
        for (int j = offset; j<range; j++){
            output[i][j] = applyMask(input, i, j);
        }
    }
    return output;
}

// __device__ double global_applyMask(double **array, int row, int col){
//     const int size = 512;
//     const int mask_size = 3;
//     const unsigned int offset = mask_size/2;
//     int n_size = offset * 2 + 1;

//     // neighbours of given location
//     // double **neighbours = allocateMatrix(n_size, n_size);
//     double **neighbours ;
//     malloc((void **) &neighbours, sizeof(double));

//     // dynamically get the neighbours
//     int n1 = 0;
//     for (int r=row - 1; r <= row + offset; r++){
//         int n2 = 0;
//         for (int c =col - 1; c <= col + offset; c++){
//             neighbours[n1][n2] = array[r][c];
//             n2++;
//         }
//         n1++;
//     }

//     // element-wise matrix multiplication
//     double **convolution = allocateMatrix(n_size, n_size);
//     double value = 0;

//     for (int r=0; r<3; r++){
//         for(int c=0; c<3; c++){
//             // printf("value: %3.6f \n", mask[1][1]);
//             convolution[r][c] = mask[r][c] * neighbours[r][c];
//             value = value + convolution[r][c];
//         }
//     }
//     // printf("value: %3.6f \n", value);
//     // printArray(convolution, offset, offset);

//     return value;
// }

__global__ void global_convolution(double **input, double **output, int offset){
    // x = get threads in x direction i.e. columns
    // y = get threads in y direction i.e. rows
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int width = gridDim.x * blockDim.x;

    for (int i = offset; i<blockDim.x; i++){
        for (int j = offset; j<blockDim.y; j++){
            // output[i][j] = global_applyMask(input, i, j);
        }
    }
}

int main(int argc, char **argv){
    int devID = findCudaDevice(0, 0);
    cudaGetDeviceProperties(0, 0);

    // const char *imageFilename = "image21.pgm";
    const char *imageFilename = "lena_bw.pgm";
    // const char *imageFilename = "man.pgm";
    // const char *imageFilename = "mandrill.pgm";

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    printArray(hData, 10, 10);

    // convert image to 2D
    double **image =  convert2D(hData, width, height);
    // printf("Input image \n");
    // printArray(image, 10, 10);

    // allocate space for padded image with zeros
    double **h_padded = allocateMatrix(padded_size, padded_size);
    h_padded = padArray(image, h_padded);
    // printf("Padded image \n");
    // printArray(padded, 10, 10);

    //-------------- Serial Convolution --------------//

    cudaEvent_t serial_start, serial_stop;
    cudaEventCreate(&serial_start);
    cudaEventCreate(&serial_stop);

    double **h_output = allocateMatrix(padded_size, padded_size);
    cudaEventRecord(serial_start);
    h_output = serial_convolution(h_padded, h_output);
    cudaEventRecord(serial_stop);
    cudaEventSynchronize(serial_stop);

    float serial_time = 0;
    cudaEventElapsedTime(&serial_time, serial_start, serial_stop);
    // printf("Convolution image \n");
    // printArray(h_output, 10, 10);

     //-------------- CUDA --------------//
    double **d_input_image;
    double **d_output_image;

    // allocate memory on device
    checkCudaErrors(cudaMalloc((void **) &d_input_image, size));
    checkCudaErrors(cudaMalloc((void **) &d_output_image, size));

    // CUDA timing of event
    cudaEvent_t global_start, global_stop, shared_start, shared_stop;
    cudaEventCreate(&global_start);
    cudaEventCreate(&global_stop);
    cudaEventCreate(&shared_start);
    cudaEventCreate(&shared_stop);

    // copy memory from host to device
    checkCudaErrors(cudaMemcpy(d_input_image, h_padded, size, cudaMemcpyHostToDevice));
    
    // dimensions for the kernel
    dim3 dimBlock (16,16);
    // dim3 dimgGrid ((width-1)/TILE_WIDTH+1, (height-1)/TILE_WIDTH+1, 1);
    dim3 dimGrid (512/(dimBlock.x),512/(dimBlock.y),1);


    //-------------- Global Convolution --------------//
    global_convolution<<<dimGrid, dimBlock>>>(d_input_image, d_output_image, offset);



    //-------------- Unpad and results --------------//
    // unpad the array
    double **unpadded = allocateMatrix(padded_size, padded_size);
    unpadded = unpad(h_output, unpadded);
    // printf("unpadded image \n");
    // printArray(unpadded, 10, 10);

    // update array
    float *result_image;
    result_image = convert1D(unpadded, width, height);

    // Write result to file
    // char outputFilename[1024];
    // strcpy(outputFilename, imagePath);
    // strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    // sdkSavePGM(outputFilename, result_image, width, height);
    // printf("Wrote '%s'\n", outputFilename);

     //-------------- CUDA Performance Metrics --------------//

    //  float serial_throughput = num_ops / (serial_time / 1000.0f) / 1000000000.0f;

     printf("Serial Convolution Time: %3.6f ms \n", serial_time);

    free(image);
    free(h_padded);
    free(h_output);
    free(unpadded);
    free(result_image);
}