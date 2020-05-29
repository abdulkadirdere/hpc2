# HPC - Assignment 2

### Code Complilation
Code can be compiled using the **make** command in the Makefile location.
You can run the following code to build serial convolution, global memory convolution and shared memory convolution respectively:
```
make serialConvolution
make globalConvolution
make sharedConvolution
```
To run the all the builds:
```
make run
```

To clean the output files:
```
make clean
```

To run all three implementations for performance metrics:
```
make convolution
```