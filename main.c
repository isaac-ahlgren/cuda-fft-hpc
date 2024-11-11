#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fft.h"

double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void gen_rand(double* arr, int n, double min, double max) {
    for (int i = 0; i < n; i++) {
        arr[i] = randfrom(min, max);
    }
}

void print_real_array(double* arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n\n");
}

void print_complex_array(complex_t* arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f+%fi ", arr[i].real, arr[i].imag);
    }
    printf("\n\n");
}

int main() {
    double rand_min = 0;
    double rand_max = 10;

    int size = 8;

    double* arr = (double*) malloc(size*sizeof(double));
    gen_rand(arr, size, rand_min, rand_max);

    //double arr[] = {1, 6, 3, 8, 9, 5, 4, 2};

    print_real_array(arr, size);

    complex_t* output = (complex_t*) malloc(size*sizeof(complex_t));

    memset(output, 0, size*sizeof(complex_t));

    fft(arr, size, output);

    print_complex_array(output, size);

    complex_t* another_output = (complex_t*) malloc(size*sizeof(complex_t));
    
    memset(another_output, 0, size*sizeof(complex_t));

    printf("starts here\n\n");
    fft_rec_wrapper(arr, size, another_output);
    print_complex_array(another_output, size);
}