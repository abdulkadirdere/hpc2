INC="./inc"
FLAGS=-I$(INC)
CC=nvcc

all: matrixTranspose vectorReduction matrixMultip run

matrixTranspose: matrixTranspose.cu
	$(CC) matrixTranspose.cu -o matrixTranspose.out

vectorReduction: vectorReduction.cu
	$(CC) vectorReduction.cu -o vectorReduction.out

matrixMultip: matrixMultip.cu
	$(CC) matrixMultip.cu -o matrixMultip.out

run:
	./matrixTranspose.out
	./vectorReduction.out
	./matrixMultip.out

clean:
	rm matrixTranspose.out vectorReduction.out matrixMultip.out
