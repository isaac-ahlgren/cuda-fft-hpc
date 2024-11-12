#ifndef _CUDA_HPP
#define _CUDA_HPP

#include "fft.h"

struct fft_instance {
    int size;
    complex_t* twiddle_factors;
    complex_t* device_twiddle_factors;
    complex_t* buf1;
    complex_t* buf2;
};

struct fft_instance* alloc_fft_instance(int size);
void free_fft_instance(struct fft_instance* fft);
void cuda_fft(float* arr, complex_t* output, struct fft_instance* fft);

#endif