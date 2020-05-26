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

#define length(arr) ((int) (sizeof (arr) / sizeof (arr)[0]))

const int size=3;
const int mask_size = 3;
const int offset = floor(mask_size/2);
const int output_size = size + 2*offset;

const int mask[3][3] = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
};

// const int mask[5][5] = {
//     {1, 1, 1, 1, 1},
//     {1, 1, 1, 1, 1},
//     {1, 1, 1, 1, 1},
//     {1, 1, 1, 1, 1},
//     {1, 1, 1, 1, 1},
// };

// const int averaging[3][3] = {
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
    int range = output_size - offset;
    // printf("%d \n", range);

    // pad the array
    for (int i = offset; i < range; i++) {
        for (int j = offset; j < range; j++) {
            output[i][j] = input[i-offset][j-offset];
        }
    }
    return output;
}

int **unpad(int **input, int **output) {
    int range = output_size - offset;

    // unpad the array
    for (int i = 0; i < range; i++) {
        for (int j = 0; j < range; j++) {
            output[i][j] = input[i+offset][j+offset];
        }
    }
    return output;
}


int applyMask(int **array, int row, int col){
    int n_size = offset * 2 + 1;

    // neighbours of giving location
    int **neighbours = createMatrix(n_size, n_size);

    int range = output_size - offset;
    // for (int i=row; i < range; i++){
    //     for(int j=col; j < range; j++){
    //         neighbours[row-offset][col-offset] = array[row-offset][col-offset];
    //     }
    // }
    // printArray(neighbours, n_size, n_size);

    neighbours[0][0] = array[row-1][col-1]; // top_left
    neighbours[0][1] = array[row-1][col]; // top_middle
    neighbours[0][2] = array[row-1][col+1]; //top_right

    neighbours[1][0] = array[row][col-1]; //middle_left
    neighbours[1][1] = array[row][col]; //middle_middle
    neighbours[1][2] = array[row][col+1]; //middle_right

    neighbours[2][0] = array[row+1][col-1]; //bottom_left
    neighbours[2][1] = array[row+1][col]; //bottom_middle
    neighbours[2][2] = array[row+1][col+1]; //bottom_right


    int **convolution = createMatrix(n_size, n_size);
    int value = 0;

    for (int r=0; r<3; r++){
        for(int c=0; c<3; c++){
            convolution[r][c] = mask[r][c] * neighbours[r][c];
            value = value + convolution[r][c];
        }
    }
    // printf("%d \n", value);
    // printArray(convolution, offset, offset);

    return value;
}

int **serial_convolution(int **input, int **output){
    int range = output_size - offset;
    // printf("%d ", range);

    for (int i = offset; i<range; i++){
        for (int j = offset; j<range; j++){
            output[i][j] = applyMask(input, i, j);
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
    printf("offset size: %d \n", offset);

    // pad the given array
    padded = padArray(input, padded);

    printArray(padded, output_size, output_size);
    printf("padded output \n");

    output = serial_convolution(padded, output);
    printArray(output, output_size, output_size);

    unpadded = unpad(output, unpadded);
    printf("unpadded output \n");
    printArray(unpadded, size, size);

}