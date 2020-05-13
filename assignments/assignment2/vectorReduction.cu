#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int randomNumberGeneration(int upperBound, int lowerBound) {
    int num = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
    return num;
}

int *createData(int *vector, int num_element) {
    for (int i = 0; i < num_element; i++) {
        vector[i] = randomNumberGeneration(9, 0);
    }
    return vector;
}

int *createVector(int num_element){
    int *vector = (int *)malloc(num_element * sizeof(int *));
    // create synthetic data for vector
    vector = createData(vector, num_element);
    return vector;
}

int *allocateVector(int num_element) {
    int *vector = (int *)malloc(num_element * sizeof(int *));
    return vector;
}

int serialVectorSum(int *h_input_vector, int num_element){
    int sum = 0;
    for (int i=0; i < num_element; i++){
        sum = sum + h_input_vector[i];
    }
    return sum;
}

void printVector(int *vector, int num_element) {
    for (int i = 0; i < num_element; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");
}

__global__ void sharedVectorSum(float *d_output, float *d_input){
    extern __shared__ int shared_data[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tdx = threadIdx.x;

    // load data to shared memory
    shared_data[tdx] = d_input[i];
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if (tdx < stride){
            shared_data[tdx] += shared_data[tdx + stride];
        }
        __syncthreads();
    }

    // thread 0 will write the results from shared memory
    if (tdx == 0){
        d_output[blockIdx.x] = shared_data[0];
    }
}

__global__ void globalVectorSum(float *d_output, float *d_input){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tdx = threadIdx.x;
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if (tdx < stride){
            d_input[i] += d_input[i + stride];
        }
        __syncthreads();
    }

    // thread 0 will write the results
    if (tdx == 0){
        d_output[blockIdx.x] = d_input[i];
    }
}

int main(){
    const int num_element = 10240000;

    // Host memory allocation
    int *h_input_vector = createVector(num_element);
    int *h_output_vector = allocateVector(num_element);

    // printVector(h_input_vector, num_element);
    int memory_space_required = num_element * sizeof(int);

    //-------------- Serial Vector Summation CPU --------------//
    cudaEvent_t serial_start, serial_stop;
    cudaEventCreate(&serial_start);
    cudaEventCreate(&serial_stop);

    cudaEventRecord(serial_start);
    int serial_sum = serialVectorSum(h_input_vector, num_element);
    cudaEventRecord(serial_stop);
    cudaEventSynchronize(serial_stop);

    float serial_time = 0;
    cudaEventElapsedTime(&serial_time, serial_start, serial_stop);
    printf("Serial Matrix Transpose Time: %3.6f ms \n", serial_time);
    // printf("Serial Sum: %d\n", serial_sum);
    cudaEventDestroy(serial_start);
    cudaEventDestroy(serial_stop);

//-------------- CUDA Vector Summation Global Memory --------------//
    // Device memory allocation
    float *d_input_vector;
    float *d_output_vector;

    cudaMalloc((void **) &d_input_vector, memory_space_required);
    cudaMalloc((void **) &d_output_vector, memory_space_required);

    // // copy memory from host to device
    cudaMemcpy(d_input_vector, h_input_vector, memory_space_required, cudaMemcpyHostToDevice);

    // CUDA timing of event
    cudaEvent_t global_start, global_stop, shared_start, shared_stop;
    cudaEventCreate(&global_start);
    cudaEventCreate(&global_stop);
    cudaEventCreate(&shared_start);
    cudaEventCreate(&shared_stop);

   // dimensions for the kernel
   int MAX_THREADS = 1024;
   int NUM_THREADS = MAX_THREADS;
   int NUM_BLOCKS = num_element / MAX_THREADS;
   // if (NUM_BLOCKS == 0 ){
   //     NUM_BLOCKS = 1;
   // }

    //-------------- CUDA Vector Summation Global Memory --------------//
    cudaEventRecord(global_start);
    globalVectorSum<<<NUM_BLOCKS, NUM_THREADS>>>(d_output_vector, d_input_vector);
    cudaEventRecord(global_stop);
    cudaEventSynchronize(global_stop);

    float global_elapsedTime = 0;
    cudaEventElapsedTime(&global_elapsedTime, global_start, global_stop);
    printf("Global Memory Time elpased: %3.6f ms \n", global_elapsedTime);
    cudaEventDestroy(global_start);
    cudaEventDestroy(global_stop);

    //-------------- CUDA Vector Summation Shared Memory --------------//
    cudaEventRecord(shared_start);
    sharedVectorSum<<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(int)>>>(d_output_vector, d_input_vector);
    cudaEventRecord(shared_stop);
    cudaEventSynchronize(shared_stop);

    float shared_elapsedTime = 0;
    cudaEventElapsedTime(&shared_elapsedTime, shared_start, shared_stop);
    printf("Shared Memory Time elpased: %3.6f ms \n", shared_elapsedTime);
    cudaEventDestroy(shared_start);
    cudaEventDestroy(shared_stop);

    cudaMemcpy(h_output_vector, d_output_vector, memory_space_required, cudaMemcpyDeviceToHost);

    // int sum = 0;
    // for (int i = 0; i<num_element; i++){
    //     sum += h_output_vector[i];
    // }
    // printf("%d \n", sum);

    free(h_input_vector);
    free(h_output_vector);
    cudaFree(d_input_vector);
    cudaFree(d_output_vector);
}