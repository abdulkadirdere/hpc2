INC="./inc"
FLAGS=-I$(INC)
CC=nvcc

all: matrixTranspose vectorReduction matrixMultip run

matrixTranspose: matrixTranspose.cu
	$(CC) matrixTranspose.cu -o matrixTranspose

vectorReduction: vectorReduction.cu
	$(CC) vectorReduction.cu -o vectorReduction

matrixMultip: matrixMultip.cu
	$(CC) matrixMultip.cu -o matrixMultip

run:
	./matrixTranspose 
	./vectorReduction
	./matrixMultip 

clean:
	rm matrixTranspose vectorReduction matrixMultip
