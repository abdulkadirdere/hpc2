#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "helper/inc/helper_cuda.h"
#include "helper/inc/helper_functions.h"

#define TILE_W 16 
#define TILE_H 16 
#define R 2 // filter radius
#define D (R*2+1) // filter diameter
#define S (D*D) // filter size
#define BLOCK_W (TILE_W+(2*R))
#define BLOCK_H (TILE_H+(2*R))
#define MASK_COLS 3 
#define MASK_ROWS 3 

// Image to perform convolution on
const char *imageFilename = "data/lena_bw.pgm";

// Loaded mask in constant memory
__constant__ float mask[MASK_ROWS*MASK_COLS];

// Global Kernel
__global__ void convolution(float* dData, float* dResult, unsigned int width, unsigned int height){

    __shared__ float smem[BLOCK_W*BLOCK_H];

    int x = blockIdx.x * TILE_W + threadIdx.x - R;
    int y = blockIdx.y * TILE_H + threadIdx.y - R;

    // Image edges
    x = max(0, x);
    x = min(x, width-1);
    y = max(y, 0);
    y = min(y, height-1);

    unsigned int index = y*width + x;
    unsigned int bindex = threadIdx.y * blockDim.y + threadIdx.x;

    smem[bindex] = dData[index];

    __syncthreads();

    // if (((threadIdx.x >= R) && (threadIdx.x < BLOCK_W-R)) && ((threadIdx.y>=R) && (threadIdx.y<=BLOCK_H-R))){
    //     float sum = 0;
    //     for(int dy=-R;dy<=R;dy++){
    //         for(int dx=-R;dx<R;dx++){
    //             float i = smem[bindex+(dy*blockDim.x)+dx];
    //             sum +=i;
    //         }
    //     }
    //     dResult[index] = sum/S;
    // }

    // dResult[index] = dData[index];
    
    if (((threadIdx.x >= R) && (threadIdx.x < BLOCK_W-R)) && ((threadIdx.y>=R) && (threadIdx.y<=BLOCK_H-R))){
        float sum = 0;
        // Iterate over mask rows
        for(int i = 0; i<MASK_ROWS; i++){
            //Iterate over mask cols
            for(int j = 0; j<MASK_COLS; j++){ 
                sum += smem[bindex+(i*blockDim.x)+j] * mask[i*3+j];
            }
        }
        dResult[index] = sum;
    }


}

int main(void){
    // Set mask in constant memory

    // Edge Filter
    // float constant_mem_mask[MASK_ROWS*MASK_COLS]= {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    // Sharpening Filter
    float constant_mem_mask[MASK_ROWS*MASK_COLS]= {-1, -1, -1, -1, 9, -1, -1, -1, -1};
    // Averaging Filter
    // float constant_mem_mask[MASK_ROWS*MASK_COLS]= {1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9};

    //Get the device properties
    int devID = findCudaDevice(0, 0);
    cudaGetDeviceProperties(0, 0);

    // Image on host
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, 0);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }
    sdkLoadPGM(imagePath, &hData, &width, &height);
    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    // Allocate space for image on device
    float *dData = NULL;
    checkCudaErrors(cudaMalloc((void **) &dData, size)); 
    checkCudaErrors(cudaMemcpy(dData, hData, size, cudaMemcpyHostToDevice));

    // Allocate memory for the resulting image on device
    float *dResult = NULL;
    checkCudaErrors(cudaMalloc((void **) &dResult, size)); 
    cudaMemcpyToSymbol(mask, &constant_mem_mask, MASK_ROWS*MASK_COLS*sizeof(float));

    // Timing using Cuda Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Perform work on GPU...
    // Set the grid and block dimensions
    // Set max threads <1024
    int threads = 32;
    // Set enough blocks equal to the DIM of your array
    int blocks = (width+threads-1)/threads;

    dim3 dimGrid(100, 100);
    dim3 dimBlock(BLOCK_W, BLOCK_H);

    convolution<<<dimGrid, dimBlock>>>(dData, dResult, height, width);

    cudaEventRecord(stop,0);
    cudaEventSynchronize( stop );
    float elapseTime;
    cudaEventElapsedTime(&elapseTime, start, stop);
    float throughput = (width*height/((elapseTime*1000)*(10^9)));

    printf( "GPU Global Mem Throughput: %3.6f ms\n", throughput);
    printf( "GPU Global Mem Time elpased: %3.6f ms\n", elapseTime );

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    //Copy the resulting image back to the host
    float *hResult = (float *)malloc(size);
    checkCudaErrors(cudaMemcpy(hResult, dResult, size, cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_GPU_out_SM.pgm");
    sdkSavePGM(outputFilename, hResult, width, height);
    printf("Wrote '%s'\n", outputFilename);

    free(hResult);
    cudaDeviceReset();
}