__kernel void vectorAdd(__global const float *a, __global const float *b,
                        __global float *result, const unsigned int size) {
  //@@ Insert code to implement vector addition here

  size_t idx = get_global_id(0);

  

  if(idx < size){
    result[idx] = a[idx] + b[idx];
    printf("Result[%d]: %f\n", idx, result[idx]);
    printf("Size: %d\n", size);
  }
}