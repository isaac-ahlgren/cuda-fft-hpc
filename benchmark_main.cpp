#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <stdint.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include <fftw3.h>
#include "fft.h"
#include "util.h"
#include "cuda_fft.hpp"



int main(int argc, char* argv[]) {

    float rand_min = 0;
    float rand_max = 10;
    uint64_t size = 64;


    if (argc == 4){
	rand_min = std::atof(argv[1]);
	rand_max = std::atof(argv[2]);
	size = std::atof(argv[3]);
    }

    std::cout << "Min: " << rand_min << ", Max: " << rand_max << ", Size: " << size << std::endl;


    float* arr = (float*) malloc(size*sizeof(float));
    gen_rand(arr, size, rand_min, rand_max);

    //float arr[] = {1, 6, 3, 8, 9, 5, 4, 2};
	
    //print_real_array(arr, size);
    
    //fftw library
    fftwf_complex* output = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(size, arr, output, FFTW_ESTIMATE);
    auto fftw_start = std::chrono::high_resolution_clock::now();
    fftwf_execute(plan);
    auto fftw_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fftw_duration = fftw_end - fftw_start;
    free(plan);
    free(output);
    std::cout << "FFTW execution time: " << fftw_duration.count() << " seconds" << std::endl;


    // Recursive FFT
    complex_t* output1 = (complex_t*) malloc(size*sizeof(complex_t));    
    memset(output1, 0, size*sizeof(complex_t));
    auto recur_start = std::chrono::high_resolution_clock::now();
    fft_rec_wrapper(arr, size, output1);
    auto recur_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> recur_duration = recur_end - recur_start;
    //print_complex_array(output1, size);
    free(output1);
    std::cout << "Recursive FFT execution time: " << recur_duration.count() << " seconds" << std::endl;

    // Iterative FFT
    complex_t* output2 = (complex_t*) malloc(size*sizeof(complex_t));
    memset(output2, 0, size*sizeof(complex_t));
    auto iter_start = std::chrono::high_resolution_clock::now();
    fft(arr, size, output2);
    auto iter_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> iter_duration = iter_end - iter_start;
    //print_complex_array(output2, size);
    free(output2);
    std::cout << "Iterative FFT execution time: " << iter_duration.count() << " seconds" << std::endl;

    
    // Cuda FFT
    struct fft_instance* fft_inst = alloc_fft_instance(size);
    complex_t* output3 = (complex_t*) malloc(size*sizeof(complex_t));
    auto cuda_start = std::chrono::high_resolution_clock::now();
    cuda_fft(arr, output3, fft_inst);
    auto cuda_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cuda_duration = cuda_end - cuda_start;
    //print_complex_array(output3, size);
    free(output3);
    free_fft_instance(fft_inst);
    std::cout << "Cuda FFT execution time: " << cuda_duration.count() << " seconds" << std::endl;

    // CuFFT
    float* cuFFT_input;
    cufftComplex* cuFFT_output;
    cudaMalloc(&cuFFT_input, size * sizeof(float));
    cudaMalloc(&cuFFT_output, sizeof(cufftComplex) * (size/2+1));

    cudaMemcpy(cuFFT_input, arr, size * sizeof(float), cudaMemcpyHostToDevice);

    cufftHandle cuFFT_plan;
    cufftPlan1d(&cuFFT_plan, size, CUFFT_R2C, 1);

    //cufftExecR2C(cuFFT_plan, cuFFT_input, cuFFT_output);

    auto cufft_start = std::chrono::high_resolution_clock::now();
    cufftExecR2C(cuFFT_plan, cuFFT_input, cuFFT_output);
    cudaDeviceSynchronize(); // Ensure GPU operations are complete
    auto cufft_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> cufft_duration = cufft_end - cufft_start;
    std::cout << "cuFFT execution time: " << cufft_duration.count() << " seconds" << std::endl;


    cudaFree(cuFFT_input);
    cudaFree(cuFFT_output);
    cufftDestroy(cuFFT_plan);

}
