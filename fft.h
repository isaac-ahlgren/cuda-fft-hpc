#ifndef FFT_H_
#define FFT_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct complex_float {
    float real;
    float imag;
} complex_t;

void fft(float*, int, complex_t*);
void fft_rec_wrapper(float*, int, complex_t*);
int bit_reverse(unsigned int n, int size);

#define TWO_PI ((float) 6.2831853071795864769252867665590057683943L)

#ifdef __cplusplus
}
#endif

#endif