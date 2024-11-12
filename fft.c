#include <math.h>
#include <stdio.h>

#include "fft.h"

int bit_reverse(unsigned int n, int size) {
    int log_n = ceil(log2f(size));
    int limit = ceil(((float)log_n / 2));

    unsigned int output = 0;
    for (int i = 0; i < limit; i++) {
        unsigned int bit_mask1 = 1 << (log_n - i - 1);
        unsigned int bit1_loc = n << (log_n - 2*i -1);
        unsigned int bit1 = bit1_loc & bit_mask1;

        unsigned int bit_mask2 = 1 << i; 
        unsigned int bit2_loc = n >> (log_n - 2*i -1);
        unsigned int bit2 = bit2_loc & bit_mask2;

        output |= bit1 | bit2;
    }
    return output;
}

void copy_to_output(float* arr, complex_t* output, int n) {
    for (int i = 0; i < n; i++) {
        unsigned int rev_i = bit_reverse(i, n);
        output[rev_i].real = arr[i];
    }
}

void fft(float *arr, int n, complex_t* output) {
    int log_n = ceil(log2f(n));

    copy_to_output(arr, output, n);

    int m = 1;
    for (int i = 0; i <= log_n; i++) {
        int m = 1 << i;
        int m2 = m >> 1;
        complex_t w_m = {cos(-TWO_PI/m), sin(-TWO_PI/m)};
        for (int j = 0; j < n; j+=m) {
            complex_t w = {1,0};
            for (int k = 0; k < m2; k++) {
                int i1 = k + j + m2;
                int i2 = k + j;

                float t_r = w.real*output[i1].real - w.imag*output[i1].imag;
                float t_i = w.real*output[i1].imag + w.imag*output[i1].real;
                complex_t t = {t_r, t_i};
                complex_t u = output[i2];

                output[i2].real = u.real + t.real; output[i2].imag = u.imag + t.imag;
                output[i1].real = u.real - t.real; output[i1].imag = u.imag - t.imag;

                //printf("w=%f + %fi\n", w.real, w.imag);
                //printf("(%f + %fi) + (%f + %fi) = %f + %fi \n", u.real, u.imag, t.real, t.imag, output[i2].real, output[i2].imag);
                //printf("(%f + %fi) - (%f + %fi) = %f + %fi \n", u.real, u.imag, t.real, t.imag, output[i1].real, output[i1].imag);

                float w_r = w.real*w_m.real - w.imag*w_m.imag;
                float w_i = w.real*w_m.imag + w.imag*w_m.real;

                w.real = w_r; w.imag = w_i;
            }
        }
        //printf("\n");
    }
}

// Can't find small bug in this one
void fft_rec(float *arr, int delta, int n, complex_t* output) {
    if (n == 1) {
        output[0].real = arr[0];
        output[0].imag = 0;
    }
    else {
        int n2 = n / 2;
        fft_rec(arr, 2*delta, n2, output);
        fft_rec(arr + delta, 2*delta, n2, output + n2);
        complex_t w = {1,0};
        complex_t w_n = {cos(-TWO_PI/n), sin(-TWO_PI/n)};
        for (int i = 0; i < n2; i++) {
            //printf("w=%f + %fi , w_m=%f + %fi, m=%d, s=%d, i=%d\n", w.real, w.imag, w_n.real, w_n.imag, n, delta, i);

            complex_t p = output[i];
            float q_r = w.real*output[i + n2].real - w.imag*output[i + n2].imag;
            float q_i = w.real*output[i + n2].imag + w.imag*output[i + n2].real; 
            complex_t q = {q_r, q_i};

            output[i].real = p.real + q.real; output[i].imag = p.imag + q.imag;
            output[i + n2].real = p.real - q.real; output[i + n2].imag = p.imag - q.imag;

            //printf("(%f + %fi) + (%f + %fi) = %f + %fi \n", p.real, p.imag, q.real, q.imag, output[i].real, output[i].imag);
            //printf("(%f + %fi) - (%f + %fi) = %f + %fi \n", p.real, p.imag, q.real, q.imag, output[i + n/2].real, output[i + n/2].imag);

            float w_r = w.real*w_n.real - w.imag*w_n.imag;
            float w_i = w.real*w_n.imag + w.imag*w_n.real;

            w.real = w_r; w.imag = w_i;
        }
        //printf("\n");
    }
}

void fft_rec_wrapper(float *arr, int n, complex_t* output) {
    fft_rec(arr, 1, n, output);
}
