#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "helper/inc/helper_cuda.h"
#include "helper/inc/helper_functions.h"

//Set default sizes of the matrix that will be generated. Perform convolution on this matrix
#define MATRIX_ROWS 5
#define MATRIX_COLS 5
//Set default sizes of the mask that will be generated. Perform convolution on this matrix
#define MASK_ROWS 3
#define MASK_COLS 3

#define OFFSET MASK_COLS/2

//For images
const char *imageFilename = "imgs/lena_bw.pgm";

//Declare the mask to be used: Averaging Filter, Sharpening and Edge Detection
//averaging_mask
// static double mask[3][3] = {{1, 1, 1},
//                             {1, 1, 1},
//                             {1, 1, 1}};

//averaging_mask
// static double mask[7][7] = {
//                                             {1, 1, 1, 1, 1, 1, 1},
//                                             {1, 1, 1, 1, 1, 1, 1},
//                                             {1, 1, 1, 1, 1, 1, 1},
//                                             {1, 1, 1, 1, 1, 1, 1},
//                                             {1, 1, 1, 1, 1, 1, 1},
//                                             {1, 1, 1, 1, 1, 1, 1},
//                                             {1, 1, 1, 1, 1, 1, 1}};

//sharpening_mask
// static double mask[MASK_ROWS][MASK_COLS] = {{-1, -1, -1, -1, -1, -1, -1, -1, -1},
//                                             {-1, -5, -5, -5, -5, -5, -5, -5, -1},
//                                             {-1, -5, -1, -9, -9, -9, -1, -5, -1},
//                                             {-1, -5, -1, -9, -9, -9, -1, -5, -1},
//                                             {-1, -5, -1, -9, -9, -9, -1, -5, -1},
//                                             {-1, -5, -5, -5, -5, -5, -5, -5, -1},
//                                             {-1, -1, -1, -1, -1, -1, -1, -1, -1}};

//edge_mask
static double mask[3][3] = {{-1, 0, 1},
                            {-2, 0, 2},
                            {-1, 0, 1}};

double **create_matrix(){

    int PADDED_MATRIX_ROWS = MATRIX_ROWS + OFFSET*2;
    int PADDED_MATRIX_COLS = MATRIX_COLS + OFFSET*2;

    //Allocates space for padded 2D array dynamically
    double *matrix = (double *)malloc(PADDED_MATRIX_ROWS * sizeof(double *));
    for (int i=0; i<PADDED_MATRIX_COLS; i++){
        matrix[i] = (double *)malloc(PADDED_MATRIX_COLS * sizeof(double));
    }

    // Generate and insert random numbers into the array created above
    for (int i=0; i<PADDED_MATRIX_ROWS; i++){
        for (int j = 0; j<PADDED_MATRIX_COLS; j++){
            // Create random integer between 0 and 9
            if ((i>=OFFSET && i<MATRIX_ROWS+OFFSET) && (j>=OFFSET && j<MATRIX_COLS+OFFSET)){
                matrix[i][j] = (rand()%9+1);
            }else{
                matrix[i][j] = 0;
            }
        }
    }
    return matrix;
}

void print_matrix(double **matrix){
    int PADDED_MATRIX_ROWS = MATRIX_ROWS + OFFSET*2;
    int PADDED_MATRIX_COLS = MATRIX_COLS + OFFSET*2;
    for (int i=0; i<PADDED_MATRIX_ROWS; i++){
        for (int j=0; j<PADDED_MATRIX_COLS; j++){
        printf("%.f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_image(double **matrix, unsigned int width, unsigned int height){
    // int PADDED_MATRIX_ROWS = height + OFFSET*2;
    // int PADDED_MATRIX_COLS = width + OFFSET*2;
    for (int i=0; i<10; i++){
        for (int j=0; j<10; j++){
        printf("%.02f ", matrix[i][j]);
        }
        printf("\n");
    }
}

double **convert_image_2D(float *hData, unsigned int width, unsigned int height){

    int PADDED_MATRIX_ROWS = height + OFFSET*2;
    int PADDED_MATRIX_COLS = width + OFFSET*2;

    //Allocates space for padded 2D array dynamically
    double *matrix = (double *)malloc(PADDED_MATRIX_ROWS * sizeof(double *));
    for (int i=0; i<PADDED_MATRIX_COLS; i++){
        matrix[i] = (double *)malloc(PADDED_MATRIX_COLS * sizeof(double));
    }

    // Generate and insert random numbers into the array created above
    int pix_value = 0;
    for (int i=0; i<PADDED_MATRIX_ROWS; i++){
        for (int j = 0; j<PADDED_MATRIX_COLS; j++){
            // Create random integer between 0 and 9
            if ((i>=OFFSET && i<height+OFFSET) && (j>=OFFSET && j<width+OFFSET)){
                matrix[i][j] = hData[pix_value];
                // matrix[i][j] = 1;
                pix_value++;
            }else{
                matrix[i][j] = 0;
            }
        }
    }
    return matrix;
}

double calculate_convoluted_pixel_value(int pixel_row, int pixel_col, double **matrix, double mask[MASK_ROWS][MASK_COLS]){
    double sum = 0;
    double multiply;
    //printf("Pixel: %f at row: %d and col: %d\n", matrix[pixel_row][pixel_col], pixel_row, pixel_col);
    int m = 0;
    for (int i=pixel_row-OFFSET; i<=pixel_row+OFFSET; i++){
        int n=0;
        for (int j = pixel_col-OFFSET; j<=pixel_col+OFFSET; j++){
            // printf("i: %d and j: %d\n", i, j);
            // printf("m: %d and n:%d\n", m, n);
            // printf("Matrix: %f\n", matrix[i][j]);
            // printf("Mask: %f\n\n", mask[m][n]);
            multiply = matrix[i][j] * mask[m][n];
            sum = sum + multiply;
            n++;
        }
        m++;
    }
    //printf("sum: %f\n", sum);
    return sum;
}

double **perform_convolution(double **matrix, double **matrix_results,double mask[MASK_ROWS][MASK_COLS]){
    for (int i=0; i<MATRIX_ROWS; i++){
        for (int j = 0; j<MATRIX_COLS; j++){
            // printf("We are operating on row: %d and col: %d\n",i+OFFSET, j+OFFSET);
            matrix_results[i+OFFSET][j+OFFSET] = calculate_convoluted_pixel_value(i+OFFSET, j+OFFSET, matrix, mask);
            // break;
        }
        // break;
    }
    return matrix_results;
}


void run_on_generated_matrix(){
    // Generate the random matrix
    double **generated_matrix =  create_matrix();
    double **matrix_results =  create_matrix();

    // Apply convolution
    printf("\n---Input Matrix---\n\n");
    print_matrix(generated_matrix);

    matrix_results = perform_convolution(generated_matrix, matrix_results,mask);

    printf("\n---Output Matrix---\n\n");
    print_matrix(matrix_results);

    free(generated_matrix);
    free(matrix_results);

}

void run_on_image(){

    // Loading of the image file
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
    
    //Covert image to 2D
    double **image =  convert_image_2D(hData, width, height);
    double **image_results =  convert_image_2D(hData, width, height);
    
    // Apply convolution
    printf("\n---Part of Input Image---\n\n");
    print_image(image, width, height);

    image_results = perform_convolution(image, image_results,mask);

    printf("\n---Output Matrix---\n\n");
    print_image(image_results, width, height);


    float *flat_image = (float *)malloc(size * sizeof(float));
    // Generate and insert random numbers into the array created above
    int k = 0;
    for (int i=0; i<height; i++){
        for (int j = 0; j<width; j++){
            flat_image[k] = (float)image_results[i][j];
            k++;
        }
    }

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, flat_image, width, height);
    printf("Wrote '%s'\n", outputFilename);

    free(image);
    free(image_results);




}

int main(void){

    clock_t t;
    t = clock();

    run_on_generated_matrix();
    run_on_image();

    t = clock() - t;
    double time_taken = (((double)t)/CLOCKS_PER_SEC)*1000;
    printf("\nTime Taken on CPU: %f\n", time_taken);
 
}