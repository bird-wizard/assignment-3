#define TILE_SIZE 16

__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {

  // We will iterate over TILE_SIZE amount of work per index
  // We will have Global Work Size / TILE_SIZE amount of indices
  // Local memory for storing sub-matrices of A and B
  __local float localA[TILE_SIZE][TILE_SIZE];
  __local float localB[TILE_SIZE][TILE_SIZE];
  
  //@@ Insert code to implement matrix multiplication here
  int row = get_global_id(0); // Row ID of C (0..M)
  int col = get_global_id(1); // Col ID of C (0..N)
  float acc = 0.0f;

  // Matrix A MxK 4x4:
  //     0  1  2  3
  // 0 [ 0  1  2  3]
  // 1 [ 4  5  6  7]
  // 2 [ 8  9 10 11]
  // 3 [12 13 14 15]

  // Matrix B KxN 4x4:
  //     0  1  2  3
  // 0 [ 0  1  2  3]
  // 1 [ 4  5  6  7]
  // 2 [ 8  9 10 11]
  // 3 [12 13 14 15]

  // Matrix C MxN 2x2:
  //     0  1
  // 0 [ 0  1]
  // 1 [ 2  3]
  
  // Matrix C Actual 4x4, TILE_SIZE = 2:
  //     0  1  2  3
  // 0 [ 0  1  2  3]
  // 1 [ 4  5  6  7]
  // 2 [ 8  9 10 11]
  // 3 [12 13 14 15]

  // C[i,j] = SUM(A[i,k]*B[k,j])
  // Iterate over columns of A up to TILE_SIZE
  // numAColumns = 4
  // TILE_SIZE = 2
  // iterate 0 and 1
  // 4 + 2 - 1 / 2 
  // iterate over tiles required to compute the current 2x2 block of C
  for (int t = 0; t < (numAColumns + TILE_SIZE - 1) / TILE_SIZE; ++t) {

      // Load the tile of A into local memory
      if (row < numARows && t * TILE_SIZE + get_local_id(0) < numAColumns) {
          localA[get_local_id(1)][get_local_id(0)] = A[row * numAColumns + t * TILE_SIZE + get_local_id(0)];
      } else {
          localA[get_local_id(1)][get_local_id(0)] = 0.0f;
      }

      // Load the tile of B into local memory
      if (col < numBColumns && t * TILE_SIZE + get_local_id(1) < numBRows) {
          localB[get_local_id(1)][get_local_id(0)] = B[(t * TILE_SIZE + get_local_id(1)) * numBColumns + col];
      } else {
          localB[get_local_id(1)][get_local_id(0)] = 0.0f;
      }

      // Synchronize to ensure all work-items have loaded their tile into local memory
      barrier(CLK_LOCAL_MEM_FENCE);

      // Perform the multiplication of the tiles
      for (int k = 0; k < TILE_SIZE; ++k) {
          acc += localA[get_local_id(1)][k] * localB[k][get_local_id(0)];
      }

      // Synchronize before loading the next tile
      barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Store the result in C
  if (row < numCRows && col < numCColumns) {
      C[row * numCColumns + col] = acc;
  }
}