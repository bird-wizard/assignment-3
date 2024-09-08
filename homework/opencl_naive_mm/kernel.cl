__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  
  //@@ Insert code to implement matrix multiplication here
  int C_row = get_global_id(0); // Row ID of C (0..M)
  int C_col = get_global_id(1); // Col ID of C (0..N)
  float acc = 0.0f;
  printf("Global ID: (%d, %d)\n",get_global_id(0), get_global_id(1));
  // Matrix A MxK 2x5:
  //    0 1 2 3 4
  // 0 [0 1 2 3 4]
  // 1 [5 6 7 8 9]

  // Matrix B KxN 5x2:
  //    0 1
  // 0 [0 1]
  // 1 [2 3]
  // 2 [4 5]
  // 3 [6 7]
  // 4 [8 9]

  // Matrix C MxN 2x2:
  //     0  1
  // 0 [ 0  1]
  // 1 [ 2  3]
  
  // C[i,j] = SUM(A[i,k]*B[k,j])
  // Iterate over columns of A
  for (int k = 0; k < numAColumns; k++) {
      acc += A[numAColumns*C_row + k] * B[k*numBColumns + C_col];
  }

  // numARows = 5
  // C[0,0] = A[0,k]*B[k,0]
  C[numCColumns*C_row + C_col] = acc;
}