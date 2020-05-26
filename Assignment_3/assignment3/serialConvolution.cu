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

const int size=3;
const int mask_size = 3;
const int output_size = size * mask_size;

const int averaging[mask_size][mask_size] = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
};

// const int averaging[mask_size][mask_size] = {
//     {2, 2, 2},
//     {2, 2, 2},
//     {2, 2, 2},
// };


void printArray(int **array, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%d ", array[i][j]);
        }
        printf("\n");
    }
}

int randomNumberGeneration(int upperBound, int lowerBound) {
    int num = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
    return num;
}

int **createMatrix(int m, int n) {
    int **array = (int **)malloc(m * sizeof(int *));
    for (int i = 0; i < m; i++) {
        array[i] = (int *)malloc(n * sizeof(int));
    }

    return array;
}

int **createData(int **array, int size, int dimension) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < dimension; j++) {
            array[i][j] = randomNumberGeneration(9, 1);
        }
    }
    return array;
}

int **padArray(int **input, int **output) {
    int range = output_size - mask_size;
    // printf("%d \n", range);

    // pad the array
    for (int i = size; i < range; i++) {
        for (int j = size; j < range; j++) {
            output[i][j] = input[i-mask_size][j-mask_size];
        }
    }
    return output;
}

int **unpad(int **input, int **output) {
    int range = output_size - mask_size;

    // unpad the array
    for (int i = 0; i < range; i++) {
        for (int j = 0; j < range; j++) {
            output[i][j] = input[i+mask_size][j+mask_size];
        }
    }
    return output;
}


int applyMask(int **array, int row, int col, const int mask[mask_size][mask_size]){
    int value = 0;
    // int range = output_size - mask_size;
    // neighbours of giving location
    int **neighbours = createMatrix(mask_size, mask_size);
    
    // for (int i=row; i < range; i++){
    //     for(int j=col; j < range; j++){
    //         neighbours[row-mask_size][col-mask_size] = array[i-1][j-1]; // array is wrong
    //     }
    // }

    neighbours[0][0] = array[row-1][col-1]; // top_left
    neighbours[0][1] = array[row-1][col]; // top_middle
    neighbours[0][2] = array[row-1][col+1]; //top_right

    neighbours[1][0] = array[row][col-1]; //middle_left
    neighbours[1][1] = array[row][col]; //middle_middle
    neighbours[1][2] = array[row][col+1]; //middle_right

    neighbours[2][0] = array[row+1][col-1]; //bottom_left
    neighbours[2][1] = array[row+1][col]; //bottom_middle
    neighbours[2][2] = array[row+1][col+1]; //bottom_right


    int **convolution = createMatrix(mask_size, mask_size);

    for (int r=0; r<mask_size; r++){
        for(int c=0; c<mask_size; c++){
            convolution[r][c] = mask[r][c] * neighbours[r][c];
            value = value + convolution[r][c];
        }
    }
    // printf("%d \n", value);
    // printArray(convolution, mask_size, mask_size);

    return value;
}

int **serial_convolution(int **input, int **output){
    int range = output_size - mask_size;
    // printf("%d ", range);

    for (int i=size; i<range; i++){
        for (int j=size; j<range; j++){
            output[i][j] = applyMask(input, i, j, averaging);
        }
    }
    return output;
}


int main(void){

    int **input = createMatrix(size,size);
    int **padded = createMatrix(output_size, output_size);
    int **output = createMatrix(output_size, output_size);
    int **unpadded = createMatrix(output_size, output_size);

    input = createData(input, size, size);
    // printArray(input, size, size);

    // pad the given array
    padded = padArray(input, padded);

    // printArray(padded, output_size, output_size);

    output = serial_convolution(padded, output);

    unpadded = unpad(output, unpadded);
    printArray(unpadded, size, size);

}