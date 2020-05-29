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
#define MASK_DIM 4
#define OFFSET (MASK_DIM/2)

#define TILE_WIDTH 16
#define RADIUS 2
#define BLOCK_WIDTH (TILE_WIDTH+(2*RADIUS))


#define DIAMETER (RADIUS*2+1) // filter diameter
#define SIZE (RADIUS*DIAMETER) // filter size

const int size=512;
const int mask_size = 5;
const int offset = floor(mask_size/2);
const int padded_size = size + 2*offset;


// allocate mask in constant memory
__constant__ float d_mask_global[MASK_DIM * MASK_DIM];
__constant__ float d_mask_shared[MASK_DIM * MASK_DIM];

const double mask[mask_size][mask_size] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
};

void printArray(double **array, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%3.6f ", array[i][j]);
        }
        printf("\n");
    }
}

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

    // dynamically get the neighbours range
    int n1 = 0;
    for (int r=row - 1; r <= row + offset; r++){
        int n2 = 0;
        for (int c =col - 1; c <= col + offset; c++){
            neighbours[n1][n2] = array[r][c];
            n2++;
        }
        n1++;
    }

    // neighbours[0][0] = array[row-1][col-1]; // top_left
    // neighbours[0][1] = array[row-1][col]; // top_middle
    // neighbours[0][2] = array[row-1][col+1]; //top_right

    // neighbours[1][0] = array[row][col-1]; //middle_left
    // neighbours[1][1] = array[row][col]; //middle_middle
    // neighbours[1][2] = array[row][col+1]; //middle_right

    // neighbours[2][0] = array[row+1][col-1]; //bottom_left
    // neighbours[2][1] = array[row+1][col]; //bottom_middle
    // neighbours[2][2] = array[row+1][col+1]; //bottom_right

    // printArray(neighbours, n_size, n_size);

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

// 2D serial convolution method
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

__global__ void shared_convolution(float* dData, float* dResult, unsigned int width, unsigned int height){

  // create tile in shared memrory for the convolution
  __shared__ float shared[BLOCK_WIDTH * BLOCK_WIDTH];

    // for simplicity to use threadIdx
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // get row and column index of pixels in the tile
    int col = bx * TILE_WIDTH + tx - RADIUS;
    int row = by * TILE_WIDTH + ty - RADIUS;

    // Find the last and first pixel locations within the image
    col = max(0, col);
    col = min(col, width-1);
    row = max(row, 0);
    row = min(row, height-1);

    // load the tile pixels from the global memory into shared memory
    // this will help us to reduce global memory access by the factor of 1/TILE_WIDTH
    // ignore any pixels which are out-of-bounds (i.e. padded area)
    unsigned int index = row * width + col;
    unsigned int block_index = ty * blockDim.y + tx;
    shared[block_index] = dData[index];

    // thread barrier to wait for all the threads to finish loading from
    // global memory to shared memory
    __syncthreads();
  
    // Elementwise multiplication of pixel and mask values and add all of the values within the mask
    // range to get output value of one pixel. Verify that we are not working out-of-bounds of the image
    // We will iterate over rows and columns within the mask dimensions (i.e. all the neighbours)
    float value = 0;
    if (((tx >= RADIUS) && (tx < BLOCK_WIDTH-RADIUS)) && ((ty>=RADIUS) && (ty<=BLOCK_WIDTH-RADIUS))){
      for(int i = 0; i<MASK_DIM; i++){
          for(int j = 0; j<MASK_DIM; j++){ 
            value += shared[block_index+(i*blockDim.x)+j] * d_mask_shared[i*3+j];
          }
      }
      dResult[index] = value;
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

      // convert image to 2D
      double **image =  convert2D(hData, width, height);
      // printf("Input image \n");
      // printArray(image, 10, 10);
  
      // allocate space for padded image
      double **padded = allocateMatrix(padded_size, padded_size);
      padded = padArray(image, padded);
      // printf("Padded image \n");
      // printArray(padded, 10, 10);

  // allocate memory for mask
  size_t mask_size = MASK_DIM * MASK_DIM * sizeof(float);

  // Allocate memory for h_result image
  float *h_result = (float *)malloc(sizeof(float) * width * height);;

  //-------------- Initialise Masks --------------//
  // edge detection
  // float h_mask[MASK_DIM][MASK_DIM] = {
  //   {-1, 0, 1},
  //   {-2, 0, 2},
  //   {-1, 0, 1},
  // };

  // shapenning filter
  // float h_mask[MASK_DIM][MASK_DIM] = {
  //   {-1, -1, -1},
  //   {-1,  9, -1},
  //   {-1, -1, -1},
  // };
//   float h_mask[MASK_DIM * MASK_DIM]= {-1, -1, -1, -1, 9, -1, -1, -1, -1};
  float h_mask[MASK_DIM * MASK_DIM] = {1,1,1,1,1,1,1,1,1, 1, 1, 1, 1, 1, 1, 1};

  // averaging filter
  // float h_mask[MASK_DIM][MASK_DIM] = {
  //   {1, 1, 1},
  //   {1, 1, 1},
  //   {1, 1, 1},
  // };

  //-------------- Serial Convolution --------------//
  cudaEvent_t serial_start, serial_stop;
  cudaEventCreate(&serial_start);
  cudaEventCreate(&serial_stop);

  double **output = allocateMatrix(padded_size, padded_size);
  cudaEventRecord(serial_start);
  output = serial_convolution(padded, output);
  cudaEventRecord(serial_stop);
  cudaEventSynchronize(serial_stop);

  float serial_time = 0;
  cudaEventElapsedTime(&serial_time, serial_start, serial_stop);
  // printf("Convolution image \n");
  // printArray(output, 10, 10);


  //-------------- CUDA --------------//
  // Allocate device memory
  float *d_image;
  float *d_result;
  checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(float) * image_size));
  checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(float) * image_size));

  // Copy data to the device
  checkCudaErrors(cudaMemcpy(d_image, hData, image_size, cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpyToSymbol(d_mask_shared, h_mask, mask_size));
  checkCudaErrors(cudaMemcpyToSymbol(d_mask_shared, h_mask, mask_size));


  // CUDA timing of event
  cudaEvent_t global_start, global_stop;
  cudaEventCreate(&global_start);
  cudaEventCreate(&global_stop);

  // Calculate grid dimensions
  int THREADS = 16;
  int BLOCKS = (width+THREADS-1)/THREADS;
    // printf("%d ", BLOCKS);
  // Dimension for the kernel launch
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

  // CUDA timing of event
  cudaEvent_t shared_start, shared_stop;
  cudaEventCreate(&shared_start);
  cudaEventCreate(&shared_stop);

  // Calculate grid dimensions for dimGrid
//   int BLOCKS = (width-1)/TILE_WIDTH+1;
  // int BLOCKS = (width+TILE_WIDTH-1)/TILE_WIDTH;
  // Dimension for the kernel launch
  // dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 dimGrid(BLOCKS, BLOCKS);
    // dim3 dimGrid(32, 32);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

  //-------------- Shared Convolution --------------//
  // start the shared memory kernel
    cudaEventRecord(shared_start);
    shared_convolution<<<dimGrid, dimBlock>>>(d_image, d_result, width, height);
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
  float num_ops= width * height * MASK_DIM * MASK_DIM; // size of the image (width * height) * size of mask (3*3)

    float serial_throughput = num_ops / (serial_time / 1000.0f) / 1000000000.0f;
    float global_throughput = num_ops / (global_elapsedTime / 1000.0f) / 1000000000.0f;
    float shared_throughput = num_ops / (shared_elapsedTime / 1000.0f) / 1000000000.0f;

    std::cout << "Matrix size: " << width << "x" << width << std::endl;
    std::cout << "Tile size: " << TILE_WIDTH << "x" << TILE_WIDTH << std::endl;

    printf("Serial Image Convolution Time: %3.6f ms \n", serial_time);
    printf("Global Memory Time elapsed: %3.6f ms \n", global_elapsedTime);
    printf( "Shared Memory Time elapsed: %3.6f ms \n", shared_elapsedTime );

    std::cout << "\nSpeedup of global memory kernel (CPU/GPU): " << serial_time / global_elapsedTime << " ms" << std::endl;
    std::cout << "Speedup of shared memory kernel (CPU/GPU): " << serial_time / shared_elapsedTime << " ms" << std::endl;
      
    std::cout << "\nThroughput of serial implementation: " << serial_throughput << " GFLOPS" << std::endl;
    std::cout << "Throughput of global memory kernel: " << global_throughput << " GFLOPS" << std::endl;
    std::cout << "Throughput of shared memory kernel: " << shared_throughput << " GFLOPS" << std::endl;
    std::cout << "Performance improvement: global over serial " <<  global_throughput / serial_throughput << "x" << std::endl;
    std::cout << "Performance improvement: shared over serial " <<  shared_throughput / serial_throughput << "x" << std::endl;
    std::cout << "Performance improvement: shared over global " <<  shared_throughput / global_throughput << "x" << std::endl;

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
