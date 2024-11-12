#ifndef UTIL_H_
#define UTIL_H_

#include "fft.h"

#ifdef __cplusplus
extern "C" {
#endif

float randfrom(float min, float max);
void gen_rand(float* arr, int n, float min, float max);
void print_real_array(float* arr, int n);
void print_complex_array(complex_t* arr, int n);

#ifdef __cplusplus
}
#endif

#endif