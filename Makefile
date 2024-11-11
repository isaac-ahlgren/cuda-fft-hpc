CC := gcc
NVCC := nvcc

all:
	$(CC) -O0 -lm -g fft.c main.c -o main 
	$(NVCC) -ccbin /usr/bin/gcc-13 hello_world.cu -o hello_world

clean:
	rm main hello_world
