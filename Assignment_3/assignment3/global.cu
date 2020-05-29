#include <cassert>
#include <cstdlib>

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

// Convolution Mask Dimension
#define MASK_DIM 3
#define OFFSET (MASK_DIM/2)

// allocate mask in constant memory
__constant__ float d_mask[MASK_DIM * MASK_DIM];

// print 1D array function
void printArray(float *array, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%3.6f ", array[(i * width) + j]);
        }
        printf("\n");
    }
}


__global__ void global_convolution(float *d_Data, float *d_result, int width, int height) {
  // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Starting index for calculation
  int i_row = row - OFFSET;
  int i_col = col - OFFSET;

  // convolution value to be calculated for each pixel's row and column
   double value = 0;
  // Iterate over all the rows
  for (int i = 0; i < MASK_DIM; i++) {
    // Go over each column
    for (int j = 0; j < MASK_DIM; j++) {
      // Range check for rowsint
      if ((i_row + i) >= 0 && (i_row + i) < height && (i_col + j) >= 0 && (i_col + j) < width) {
        // Range check for columns
        // if ((start_col + j) >= 0 && (start_col + j) < width) {
        //   printf("martix %d x %d value: %3.6 --- Mask value: %3.6f \n",i,j, matrix[(start_row + i) * N + (start_col + j)], d_mask[i * MASK_DIM + j]);
            value += d_Data[(i_row + i) * width + (i_col + j)] * d_mask[i * MASK_DIM + j];
        // }
      }
    }
  }
  // write back convolution result
  d_result[row * width + col] = value;
}


// Initializes an n x n matrix with random numbers
// Takes:
//  m : Pointer to the matrix
//  n : Dimension of the matrix (square)
void init_matrix(float *m, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[n * i + j] = 1;
    }
  }
}


void verify_result(float *m, float *d_mask, float *result, int N) {
  // Temp value for accumulating results
  float temp;

  // Intermediate value for more readable code
  int offset_r;
  int offset_c;

  // Go over each row
  for (int i = 0; i < N; i++) {
    // Go over each column
    for (int j = 0; j < N; j++) {
      // Reset the temp variable
      temp = 0;

      // Go over each mask row
      for (int k = 0; k < MASK_DIM; k++) {
        // Update offset value for row
        offset_r = i - OFFSET + k;

        // Go over each mask column
        for (int l = 0; l < MASK_DIM; l++) {
          // Update offset value for column
          offset_c = j - OFFSET + l;

          // Range checks if we are hanging off the matrix
          if (offset_r >= 0 && offset_r < N) {
            if (offset_c >= 0 && offset_c < N) {
              // Accumulate partial results
              temp += m[offset_r * N + offset_c] * d_mask[k * MASK_DIM + l];
            }
          }
        }
      }
      // Fail if the results don't match
      assert(result[i * N + j] == temp);
    }
  }
}

int main(int argc, char **argv){
  // Dimensions of the matrix (2 ^ 10 x 2 ^ 10)
//   int N = 512;
    int N = 1<<9;
    // printf("%d \n",N);

//   // Size of the matrix (in bytes)
//   size_t image_size = N * N * sizeof(float);


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

    // printf("Input image \n");
    // printArray(hData, 10);

  // Size of the mask in bytes
  // Size of the matrix (in bytes)
  size_t image_size = width * height * sizeof(float);
  size_t mask_size = MASK_DIM * MASK_DIM * sizeof(float);
    // Allocate the matrix and initialize it
    float *h_result = (float *)malloc(sizeof(float) * width * height);;

  // Allocate the mask and initialize it
//   float *h_mask = new float[MASK_DIM * MASK_DIM];
//   init_matrix(h_mask, MASK_DIM);
  float h_mask[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
//   printArray(h_mask, 7);

 //-------------- CUDA --------------//
  // Allocate device memory
  float *d_image;
  float *d_result;
  checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(float) * image_size));
  checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(float) * image_size));

  // Copy data to the device
  checkCudaErrors(cudaMemcpy(d_image, hData, image_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(d_mask, h_mask, mask_size));

  // CUDA timing of event
  cudaEvent_t global_start, global_stop, shared_start, shared_stop;
  cudaEventCreate(&global_start);
  cudaEventCreate(&global_stop);
  cudaEventCreate(&shared_start);
  cudaEventCreate(&shared_stop);


  // Calculate grid dimensions
  int THREADS = 16;
  int BLOCKS = (N+THREADS-1)/THREADS;
    // printf("%d ", BLOCKS);
  // Dimension launch arguments
  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS, BLOCKS);

  //-------------- Global Convolution --------------//
  // start the global memory kernel
  cudaEventRecord(global_start);
  global_convolution<<<grid_dim, block_dim>>>(d_image, d_result, width, height);
  cudaEventRecord(global_stop);
  cudaEventSynchronize(global_stop);

  float global_elapsedTime = 0;
  cudaEventElapsedTime(&global_elapsedTime, global_start, global_stop);
  cudaEventDestroy(global_start);
  cudaEventDestroy(global_stop);

  // Copy the h_result back to the CPU
  checkCudaErrors(cudaMemcpy(h_result, d_result, image_size, cudaMemcpyDeviceToHost));
//   printf("Result image \n");
//   printArray(h_result, 10);

  // Functional test
    //  verify_result(hData, h_mask, h_result, N);
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, h_result, width, height);
    printf("Wrote '%s'\n", outputFilename);

  //-------------- CUDA Performance Metrics --------------//
  printf("Global Memory Time elpased: %3.6f ms \n", global_elapsedTime);

  // Free the memory we allocated
  free(imagePath);
  free(h_result);

//   checkCudaErrors(cudaFree(h_mask));
//   checkCudaErrors(cudaFree(d_mask));
  checkCudaErrors(cudaFree(d_image));
  checkCudaErrors(cudaFree(d_result));

  return 0;
}
