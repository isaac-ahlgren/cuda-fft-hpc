#ifndef FFT_H_
#define FFT_H_

typedef struct complex_double {
    double real;
    double imag;
} complex_t;

void fft(double*, int, complex_t*);
void fft_rec_wrapper(double*, int, complex_t*);

#endif