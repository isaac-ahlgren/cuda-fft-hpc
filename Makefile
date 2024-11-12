NVCC := nvcc

all:
	$(NVCC) -lm -ccbin /usr/bin/gcc-13 util.c fft.c cuda_fft.cu main.cpp -o main

nvcc_dump:
	$(NVCC) -ccbin /usr/bin/gcc-13 -ptx cuda_fft.cu -o cuda_fft.ptx

clean:
	rm main cuda_fft.ptx
