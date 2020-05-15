INC="./inc"
FLAGS=-I$(INC)
CC=nvcc

all: matrixTranspose vectorReduction matrixMultiplication run clean

matrixTranspose: matrixTranspose.cu
	$(CC) matrixTranspose.cu -o matrixTranspose.out

vectorReduction: vectorReduction.cu
	$(CC) vectorReduction.cu -o vectorReduction.out

matrixMultiplication: matrixMultiplication.cu
	$(CC) matrixMultiplication.cu -o matrixMultiplication.out

run:
	srun -N1 ./matrixTranspose.out
	srun -N1 ./vectorReduction.out
	srun -N1 ./matrixMultiplication.out

clean:
	rm matrixTranspose.out vectorReduction.out matrixMultiplication.out
