CC := gcc
NVCC := nvcc

all:
	$(CC) -O0 -lm -g util.c fft.c main.c -o main 
	$(NVCC) -lm -ccbin /usr/bin/gcc-13 util.c fft.c cuda_fft.cu  -o cuda_main

nvcc_dump:
	$(NVCC) -ccbin /usr/bin/gcc-13 -ptx cuda_fft.cu -o cuda_fft.ptx

clean:
	rm main cuda_main cuda_fft.ptx
