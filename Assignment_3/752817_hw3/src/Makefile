INC="./inc"
FLAGS=-I$(INC)
CC=nvcc

all: serialConvolution globalConvolution sharedConvolution run clean

serialConvolution: serialConvolution.cu
	$(CC) serialConvolution.cu -o serialConvolution.out
	srun -N1 ./serialConvolution.out

globalConvolution: globalConvolution.cu
	$(CC) globalConvolution.cu -o globalConvolution.out
	srun -N1 ./globalConvolution.out

sharedConvolution: sharedConvolution.cu
	$(CC) sharedConvolution.cu -o sharedConvolution.out
	srun -N1 ./sharedConvolution.out

convolution: convolution.cu
	$(CC) convolution.cu -o convolution.out
	srun -N1 ./convolution.out
	rm convolution.out

run:
	srun -N1 ./serialConvolution.out
	srun -N1 ./globalConvolution.out
	srun -N1 ./sharedConvolution.out
	# ./serialConvolution.out
	# ./globalConvolution.out
	# ./sharedConvolution.out

clean:
	rm serialConvolution.out globalConvolution.out sharedConvolution.out
