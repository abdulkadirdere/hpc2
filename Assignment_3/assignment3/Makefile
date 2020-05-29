INC="./inc"
FLAGS=-I$(INC)
CC=nvcc

all: serialConvolution globalConvolution sharedConvolution run clean

serialConvolution: serialConvolution.cu
	$(CC) serialConvolution.cu -o serialConvolution.out

globalConvolution: globalConvolution.cu
	$(CC) globalConvolution.cu -o globalConvolution.out

sharedConvolution: sharedConvolution.cu
	$(CC) sharedConvolution.cu -o sharedConvolution.out

run:
	# srun -N1 ./serialConvolution.out
	# srun -N1 ./globalConvolution.out
	# srun -N1 ./sharedConvolution.out
	./serialConvolution.out
	./globalConvolution.out
	./sharedConvolution.out

clean:
	rm serialConvolution.out globalConvolution.out sharedConvolution.out
