#ifndef FFT_H_
#define FFT_H_

#include <stdint.h>

#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif


void fft(float*, uint64_t, complex_t*);
void fft_rec_wrapper(float*, uint64_t, complex_t*);
uint64_t bit_reverse(uint64_t n, int size);

#define TWO_PI ((float) 6.2831853071795864769252867665590057683943L)

#ifdef __cplusplus
}
#endif

#endif
