// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

#define N 256

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( err != cudaSuccess) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) {
    // pointers for host memory and size
    int *h_a = NULL, *h_b = NULL;
    // pointers for device memory
    int *d_a = NULL;

    // YOUR CODE GOES HERE
    // Allocate memory for N integers on the host for h_a and h_b
    // (a)
   


    // YOUR CODE GOES HERE
    // Allocate memory for N integers on the device for d_a
    // (b)
    


    // Initialize h_a to contain integers 0 .. N - 1
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
    }

    // Zero memory for h_b
    memset(h_b, 0, N * sizeof(int));

    // YOUR CODE GOES HERE
    // Transfer contents of h_a to to d_a
    // (c)
   

    // YOUR CODE GOES HERE
    // Transfer contents of d_a to h_b
    // (d)
  

    // Check for any CUDA errors
    checkCUDAError("memcpy");

    for (int i = 0; i < N; i++) {
        if (h_a[i] != h_b[i]) {
            printf("Test failed h_a[%d] != h_b[%d]\n", i, i);
            exit(1);
        }
    }

    // YOUR CODE GOES HERE
    // Free memory for host pointers h_a and h_b
    // (e)
    

    // YOUR CODE GOES HERE
    // Free memory for device pointer d_a
    // (f)
   
    printf("Test passed!\n");

    return 0;
}
