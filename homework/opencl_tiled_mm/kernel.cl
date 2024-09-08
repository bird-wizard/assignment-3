#define TILE_SIZE 16

__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {

  // We will iterate over TILE_SIZE amount of work per index
  //@@ Insert code to implement matrix multiplication here

  // Get the work-item indices
  int global_row = get_global_id(0);
  int global_col = get_global_id(1); 
  //int global_row = TILE_SIZE*get_group_id(0) + get_local_id(0);
  //int global_col = TILE_SIZE*get_group_id(1) + get_local_id(1);

  // Same with Local ID
  int local_row = get_local_id(0);
  int local_col = get_local_id(1);

  // Allocate shared memory (local memory in OpenCL) for the tile of A and B
  __local float local_A[TILE_SIZE][TILE_SIZE];
  __local float local_B[TILE_SIZE][TILE_SIZE];

  // Accumulate the result of matrix multiplication in a register
  float sum = 0.0f;
  int num_tiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;
  
  // Loop over tiles (or "Group ID")
  for (int t = 0; t < num_tiles; t++) {
    // Load tile elements of A and B into local memory
    // Get global index for A with respect to tile index (Iterate across the row)
    int tiled_row = TILE_SIZE*t + local_row;
    int tiled_col = TILE_SIZE*t + local_col;

    if(global_row < numARows && tiled_col < numAColumns){
      local_A[local_row][local_col] = A[global_row*numAColumns + tiled_col];
    }
    else{
      local_A[local_row][local_col] = 0.0f;
    }
    
    if(global_col < numBColumns && tiled_row < numBRows){
      local_B[local_row][local_col] = B[tiled_row*numBColumns + global_col];
    }
    else{
      local_B[local_row][local_col] = 0.0f;
    }
    
    // Synchronize to ensure all work-items have loaded their elements
    barrier(CLK_LOCAL_MEM_FENCE);

    // Multiply the two tiles
    for (int i = 0; i < TILE_SIZE; i++) {
        sum += local_A[local_row][i] * local_B[i][local_col];
    }

    // Synchronize before loading the next tile
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write the final result to global memory
  if(global_row < numCRows && global_col < numCColumns){
    C[global_row * numCColumns + global_col] = sum;
  }
  
}