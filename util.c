#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fft.h"

float randfrom(float min, float max) 
{
    float range = (max - min); 
    float div = RAND_MAX / range;
    return min + (rand() / div);
}

void gen_rand(float* arr, int n, float min, float max) {
    for (int i = 0; i < n; i++) {
        arr[i] = randfrom(min, max);
    }
}

void print_real_array(float* arr, int n) {
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