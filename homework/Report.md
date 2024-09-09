# Assignment 3

## Lab 3

### POCL Debugging
The POCL lab introduced print statements into the kernel.cl that allowed me to print out global and local IDs to help me understand their function.

## Homework
### OpenCL Naive Matrix Multiplication
The difficult part of this assignment was understanding which global IDs incremented first and how that affected the kernel algorithm approach.
Once I figured out how the global ID iterated over the resulting matrix, I set the algorithm to iterate over the A row and the B column. 

### OpenCL Tiled Matrix Multiplication
The difficult part of this assignment was understanding how tiling and local memory worked when iterating over a set of the resulting matrix. 
Loading to local memory made sense, but it was the multiplication of the local matrices that didn't make sense. 
The pseudocode helped a ton for getting the code to work, afterwards, I spent some time making sense of the local memory and the memory fences.
I had to refer back to the openCL memory model to understand that the local memory was in fact completing all necessary loading and calculations of the two input matrices. 
The resulting sum and matrix were being pulled together in the global memory in a sense.