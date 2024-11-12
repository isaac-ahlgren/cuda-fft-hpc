#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fft.h"
#include "util.h"

int main() {
    float rand_min = 0;
    float rand_max = 10;

    int size = 8;

    float* arr = (float*) malloc(size*sizeof(float));
    gen_rand(arr, size, rand_min, rand_max);

    //float arr[] = {1, 6, 3, 8, 9, 5, 4, 2};

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