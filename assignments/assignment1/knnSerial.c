#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int randomNumberGeneration(int upperBound, int lowerBound) {
    //random number generation
    int num = (rand() % (upperBound - lowerBound + 1)) + lowerBound;
    // printf("%i\n", num);
    return num;
}

void printArray(double **array, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%f ", array[i][j]);
        }
        printf("\n");
    }
}

double **createMatrix(int m, int n) {
    double **array = (double **)malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        array[i] = (double *)malloc(n * sizeof(double));
    }

    return array;
}

double **eucledianDistance(double **reference, double **query, int mRef, int nQuery, int dimension) {
    double **eucledian = createMatrix(nQuery, mRef);
    for (int r = 0; r < nQuery; r++) {
        for (int c = 0; c < mRef; c++) {
            double sum = 0;
            for (int dim = 0; dim < dimension; dim++) {
                sum += pow(reference[c][dim] - query[r][dim], 2);
            }
            eucledian[r][c] = sqrt(sum);
        }
    }
    return eucledian;
}

double **manhattanDistance(double **reference, double **query, int mRef, int nQuery, int dimension) {
    double **manhattan = createMatrix(nQuery, mRef);
    for (int r = 0; r < nQuery; r++) {
        for (int c = 0; c < mRef; c++) {
            double sum = 0;
            for (int dim = 0; dim < dimension; dim++) {
                sum += abs(reference[c][dim] - query[r][dim]);
            }
            manhattan[r][c] = sum;
        }
    }
    return manhattan;
}

void bubbleSort(double array[], int size) {
    for (int step = 0; step < size - 1; step++) {
        for (int i = 0; i < size - step - 1; i++) {
            if (array[i] > array[i + 1]) {
                double temp = array[i];
                array[i] = array[i + 1];
                array[i + 1] = temp;
            }
        }
    }
}

void swap(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

int partition(double array[], int low, int high) {
    double pivot = array[high];
    int i = (low - 1);

    for (int j = low; j < high; j++) {
        if (array[j] <= pivot) {
            i++;
            swap(&array[i], &array[j]);
        }
    }
    swap(&array[i + 1], &array[high]);
    return (i + 1);
}

void quickSort(double array[], int low, int high) {
    if (low < high) {
        int pi = partition(array, low, high);
        quickSort(array, low, pi - 1);
        quickSort(array, pi + 1, high);
    }
}

void selectionSort(double array[], int size) {
    for (int step = 0; step < size - 1; step++) {
        int min_index = step;
        for (int i = step + 1; i < size; i++) {
            if (array[i] < array[min_index])
                min_index = i;
        }
        swap(&array[min_index], &array[step]);
    }
}

void merge(double arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    /* create temp arrays */
    double L[n1], R[n2];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++) {
        L[i] = arr[l + i];
    }

    for (j = 0; j < n2; j++) {
        R[j] = arr[m + 1 + j];
    }

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0;  // Initial index of first subarray
    j = 0;  // Initial index of second subarray
    k = l;  // Initial index of merged subarray
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there 
       are any */
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there 
       are any */
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

/* l is for left index and r is right index of the 
   sub-array of arr to be sorted */
void mergeSort(double arr[], int l, int r) {
    if (l < r) {
        // Same as (l+r)/2, but avoids overflow for
        // large l and h
        int m = l + (r - l) / 2;

        // Sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}


int main() {
    //seed
    srand(0);

    // int m[] = {200, 400};              // reference points
    // int n[] = {200, 400, 800, 1600};   // query points
    // int d[] = {32, 64, 128, 256, 512}; // dimensions in R^d

    int mRef = 100000;   // referance points
    int nQuery = 50;    // n[0]; // query points
    int dimension = 20;  // d[0];

    // double **refPoints = (double **)malloc(dimension * mRef * sizeof(double));
    // double **quryPoints = (double **)malloc(dimension * nQuery * sizeof(double));

    //allocate memory for the arrays
    double **refPoints = createMatrix(mRef, dimension);
    // double **refPoints = (double **)malloc(mRef * sizeof(double *));
    // for (int i = 0; i < mRef; i++) {
    //     refPoints[i] = (double *)malloc(dimension * sizeof(double));
    // }

    double **queryPoints = createMatrix(nQuery, dimension);
    // double **queryPoints = (double **)malloc(mRef * sizeof(double *));
    // for (int i = 0; i < nQuery; i++) {
    //     queryPoints[i] = (double *)malloc(dimension * sizeof(double));
    // }

    // synthetic data for reference points
    for (int i = 0; i < mRef; i++) {
        for (int j = 0; j < dimension; j++) {
            refPoints[i][j] = randomNumberGeneration(10, 1);
        }
    }

    // synthetic data for query points
    for (int i = 0; i < nQuery; i++) {
        for (int j = 0; j < dimension; j++) {
            queryPoints[i][j] = randomNumberGeneration(10, 1);
        }
    }

    // printf("queryPoints \n");
    // printArray(queryPoints, nQuery, dimension);

    // printf("refPoints \n");
    // printArray(refPoints, mRef, dimension);

    // calculate the distance

    double **euclidean = eucledianDistance(refPoints, queryPoints, mRef, nQuery, dimension);
    // double **manhattan = manhattanDistance(refPoints, queryPoints, mRef, nQuery, dimension);
    // printf("Distance \n");
    // printArray(euclidean, nQuery, mRef);
    // printArray(manhattan, nQuery, mRef);

    for (int i = 0; i < nQuery; i++) {
        // bubbleSort(euclidean[i], mRef);
        // selectionSort(euclidean[i],mRef);
        quickSort(euclidean[i], 0, mRef-1);
        // mergeSort(euclidean[i], 0, mRef-1);
    }

    // for (int i = 0; i < nQuery; i++) {
    //     // bubbleSort(manhattan[i], mRef);
    //     // selectionSort(manhattan[i],mRef);
    //     // quickSort(manhattan[i], 0, mRef-1);
    //     // mergeSort(manhattan[i], 0, mRef-1);
    // }

    // printf("Distance after \n");
    // printArray(euclidean, nQuery, mRef);
    // printArray(manhattan, nQuery, mRef);

    return 0;
}

