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

#define TILE_WIDTH 12
#define BLOCK_WIDTH (TILE_WIDTH + MASK_DIM -1)


// allocate mask in constant memory
__constant__ float d_mask_global[MASK_DIM * MASK_DIM];
__constant__ float d_mask_shared[MASK_DIM][MASK_DIM];

// print 1D array function
void printArray(float *array, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%3.6f ", array[(i * width) + j]);
        }
        printf("\n");
    }
}

// 2D convolution using global and constant memory
__global__ void global_convolution(float *d_Data, float *d_result, int width, int height) {
  // calculate the row and column index to compute for each thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Starting index for convolution so we can ignore the padded area
  int i_row = row - OFFSET;
  int i_col = col - OFFSET;

  // convolution value to be calculated for each pixel's row and column
  double value = 0;
  // iterate over all rows and column using the mask dimension.
  // this will calulate all the neighbours and origin pixel and sum these values to give
  // us the value of the origin pixel
  for (int i = 0; i < MASK_DIM; i++) {
    for (int j = 0; j < MASK_DIM; j++) {
      if ((i_row + i) >= 0 && (i_row + i) < height && (i_col + j) >= 0 && (i_col + j) < width) {
        //   printf("martix %d x %d value: %3.6 --- Mask value: %3.6f \n",i,j, matrix[(start_row + i) * N + (start_col + j)], d_mask[i * MASK_DIM + j]);
        value += d_Data[(i_row + i) * width + (i_col + j)] * d_mask_global[i * MASK_DIM + j];
      }
    }
  }
  // write back convolution result
  d_result[row * width + col] = value;
}

// 2D convolution using shared and constant memory
__global__ void shared_conv(float *d_data, float *d_result, unsigned int width, unsigned int height) {
  
  // create tile in shared memrory for the convolution
  __shared__ float shared[TILE_WIDTH + MASK_DIM -1][TILE_WIDTH + MASK_DIM -1];
  
  // for simplicity to use threadIdx
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // get row and column index of pixels in the tile
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  
  // row and column index to stat from so we can ignore the padded area.
  int row_i = row - OFFSET;
  int col_i = col - OFFSET;

  // __syncthreads();
  // load the tile pixels from the global memory into shared memory
  // this will help us to reduce global memory access by the factor of 1/TILE_WIDTH
  // ignore any pixels which are out-of-bounds (i.e. padded area)
  if ((row_i>=0) && (row_i < height) && (col_i >=0) && (col_i < width)){
    shared[ty][tx] = d_data[row_i*width+col_i];
  } else {
    shared[ty][tx]=0;
  }

  // thread barrier to wait for all the threads to finish loading from
  // global memory to shared memory
  __syncthreads();

  float output =0;
  // only certain threads calculate the result
  // Elementwise multiplication of pixel and mask values and add all of the neighbours
  // to get output of one pixel (origin pixel)
  if (ty < TILE_WIDTH &&  tx < TILE_WIDTH){
    for (int i=0; i< MASK_DIM; i++){
      for (int j=0; j<MASK_DIM; j++){
        output += d_mask_shared[i][j] * shared[i+ty][j+tx];
      }
    }
  }

  // thread barrier to wait for all threads to finish convolution
  __syncthreads();

  // write output to the results image
  if (row < height && col < width){
    d_result[row * width + col] = output;
  }
}



int main(int argc, char **argv){

  // image file names as input
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

  unsigned int image_size = width * height * sizeof(float);
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

  // printf("Input image \n");
  // printArray(hData, 10);

  // allocate memory for mask
  size_t mask_size = MASK_DIM * MASK_DIM * sizeof(float);

  // Allocate memory for h_result image
  float *h_result = (float *)malloc(sizeof(float) * width * height);;

  //-------------- Initialise Masks --------------//
  // edge detection
  float h_mask[MASK_DIM][MASK_DIM] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
  };

  // shapenning filter
  // float h_mask[MASK_DIM][MASK_DIM] = {
  //   {-1, -1, -1},
  //   {-1,  9, -1},
  //   {-1, -1, -1},
  // };

  // averaging filter
  // float h_mask[MASK_DIM][MASK_DIM] = {
  //   {1, 1, 1},
  //   {1, 1, 1},
  //   {1, 1, 1},
  // };

  //-------------- CUDA --------------//
  // Allocate device memory
  float *d_image;
  float *d_result;
  checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(float) * image_size));
  checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(float) * image_size));

  // Copy data to the device
  checkCudaErrors(cudaMemcpy(d_image, hData, image_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(d_mask_shared, h_mask, mask_size));

  // CUDA timing of event
  cudaEvent_t shared_start, shared_stop;
  cudaEventCreate(&shared_start);
  cudaEventCreate(&shared_stop);

  // Calculate grid dimensions for dimGrid
  int BLOCKS = (width-1)/TILE_WIDTH+1;
  // int BLOCKS = (width+TILE_WIDTH-1)/TILE_WIDTH;
  // Dimension for the kernel launch
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 dimGrid(BLOCKS, BLOCKS);

  //-------------- Shared Convolution --------------//
  // start the shared memory kernel
    cudaEventRecord(shared_start);
    shared_conv<<<dimGrid, dimBlock>>>(d_image, d_result, width, height);
    cudaEventRecord(shared_stop); 
    cudaEventSynchronize(shared_stop);
        
    float shared_elapsedTime = 0;
    cudaEventElapsedTime(&shared_elapsedTime, shared_start, shared_stop);

    cudaEventDestroy(shared_start);
    cudaEventDestroy(shared_stop);

    // Copy the h_result back to the CPU
    checkCudaErrors(cudaMemcpy(h_result, d_result, image_size, cudaMemcpyDeviceToHost));
    //   printf("Result image \n");
    //   printArray(h_result, 10);

  //-------------- Write Convolution Results to output image --------------//
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_shared.pgm");
  sdkSavePGM(outputFilename, h_result, width, height);
  printf("Wrote '%s'\n", outputFilename);

  //-------------- CUDA Performance Metrics --------------//
  float num_ops= width * height; // every element swap once

  float shared_throughput = num_ops / (shared_elapsedTime / 1000.0f) / 1000000000.0f;
  
  std::cout << "Matrix size: " << width << "x" << height << std::endl;
  std::cout << "Tile size: " << TILE_WIDTH << "x" << TILE_WIDTH << std::endl;

  printf("Shared Memory Time elpased: %3.6f ms \n", shared_elapsedTime);

  std::cout << "Throughput of shared memory kernel: " << shared_throughput << " GFLOPS" << std::endl;





  //-------------- CUDA Free Memory --------------//
  // Free the memory we allocated
  free(imagePath);
  free(h_result);

  //   checkCudaErrors(cudaFree(h_mask));
  //   checkCudaErrors(cudaFree(d_mask));
  checkCudaErrors(cudaFree(d_image));
  checkCudaErrors(cudaFree(d_result));

  return 0;
}
