#include <iostream>
#include <fftw3.h>
#include <chrono>
#include <vector>
#include <cmath>

int main (int argc, char* argv[]){
	int N = 1024;


	if (argc > 1){
		N = std::atoi(argv[1]);
	}

	std::cout << "FFT size (N): " << N << std::endl;

	std::vector<double> input(N, 0.0);
	for(int i = 0; i < N; ++i){
		input[i] = sin(2 * M_PI * i/N);
	}

	fftw_complex* output = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	fftw_plan plan = fftw_plan_dft_r2c_1d(N, input.data(), output, FFTW_ESTIMATE);

	auto start = std::chrono::high_resolution_clock::now();
	fftw_execute(plan);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = end-start;
	std::cout <<"FFT execution time: " << elapsed.count() << " secpmds" <<std::endl;

	fftw_destroy_plan(plan);
	fftw_free(output);

	return 0;
}

