#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fft.h"
#include "cuda_fft.hpp"
#include "util.h"

int main() {
    float rand_min = 0;
    float rand_max = 10;

    int size = 64;

    float* arr = (float*) malloc(size*sizeof(float));
    gen_rand(arr, size, rand_min, rand_max);

    //float arr[] = {1, 6, 3, 8, 9, 5, 4, 2};

    print_real_array(arr, size);

    // Recursive FFT
    complex_t* output1 = (complex_t*) malloc(size*sizeof(complex_t));    
    memset(output1, 0, size*sizeof(complex_t));
    fft_rec_wrapper(arr, size, output1);
    print_complex_array(output1, size);
    free(output1);

    // Iterative FFT
    complex_t* output2 = (complex_t*) malloc(size*sizeof(complex_t));
    memset(output2, 0, size*sizeof(complex_t));
    fft(arr, size, output2);
    print_complex_array(output2, size);
    free(output2);

    // Cuda FFT
    struct fft_instance* fft_inst = alloc_fft_instance(size);
    complex_t* output3 = (complex_t*) malloc(size*sizeof(complex_t));
    cuda_fft(arr, output3, fft_inst);
    print_complex_array(output3, size);
    free(output3);
    free_fft_instance(fft_inst);
}