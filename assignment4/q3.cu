#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>

#define INPUT_FILE "inp.txt"
#define Q3_OUT_FILE "q3.txt"

typedef struct vector {
  int *elements;
  int capacity;
  int size;
} vector;

// Method definitions
void int_vector_init(vector *vector);
int int_vector_add(vector* vector, int element);
void int_vector_free(vector *vector);

int chopString(char *buf, size_t size);

__global__ void get_odd_array(int* d_out, int* d_in, int size) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;

  if (myId < size) {
    int f = d_in[myId];
    d_out[myId] = f % 2;
  } 
}

__global__ void parallel_prefix_kernel(int * d_out, int * d_in, int size) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int d = 1;
  
  while(d < size) {
    if(myId+1 > d && myId < size) {
      d_in[myId] += d_in[myId - d];
    }
    d *= 2;
    __syncthreads();
  }
  
  d_out[myId] = d_in[myId];
}

__global__ void move_odds(int* d_out, int* d_in, int* prefix, int size) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;

  if (myId < size) {
    int index = prefix[myId] - 1;
    
    if (d_in[myId] % 2 == 1) {
      d_out[index] = d_in[myId];
    }
  }
}

int main(int argc, char ** argv) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }
  
  int dev = 0;
  cudaSetDevice(dev);
  
  // Read in file
  FILE *fp;
  if((fp = fopen(INPUT_FILE, "r")) == 0) {
    printf("%s cannot be found\n", INPUT_FILE);
    exit(-1);
  }
  
  char separators[] = " ,";
  char number[7];
  char buf[100];
  char* token;
  int offset = 0;
  vector *a = (vector*) malloc(sizeof(vector));
  int_vector_init(a);

  while(fgets(buf + offset, sizeof buf - offset, fp) != NULL) {
    //chop off number from string if it ends with digit
    offset = chopString(buf, sizeof buf);
    int indexOfLastNum = sizeof buf - offset - 1;// -1 to not copy '\0'

    //printf("buffer: %s\n", buf);  
  
    token = strtok(buf, separators);
    while (token != NULL) {
      int num = atoi(token);
      //printf("%d\n", num); 
      int_vector_add(a, num);
      token = strtok(NULL, separators);
    }

    memcpy(buf, &buf[indexOfLastNum], offset); 
  }
  const int ARRAY_BYTES = sizeof(int) * a->size;

  // declare GPU memory pointers
  int* d_in, * d_ones, * d_prefix, * d_out;

  // allocate GPU memory
  cudaMalloc((void**) &d_in, ARRAY_BYTES);
  cudaMalloc((void**) &d_ones, ARRAY_BYTES);
  cudaMalloc((void**) &d_prefix, ARRAY_BYTES);
  cudaMalloc((void**) &d_out, ARRAY_BYTES);

  // transfer the array to the GPU
  cudaMemcpy(d_in, a->elements, ARRAY_BYTES, cudaMemcpyHostToDevice);

  // kernels
  const int maxThreadsPerBlock = 1024; //increased from 512 to 1024 to handle 1024^2 values
  int threads = maxThreadsPerBlock;
  int blocks = (a->size + (maxThreadsPerBlock-1)) / maxThreadsPerBlock;

  // STEP 1: ones array
  get_odd_array<<<blocks, threads>>>(d_ones, d_in, a->size);

  // STEP 2: parallel prefix sum
  parallel_prefix_kernel<<<blocks, threads>>>(d_prefix, d_ones, a->size);

  // STEP 3: move odds into out using the prefix (gives the index to move to)
  move_odds<<<blocks, threads>>>(d_out, d_in, d_prefix, a->size);

  // copy back the result array to the CPU
  int h_out[a->size];
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  // write to file
  FILE *outfile = fopen(Q3_OUT_FILE, "w");
  if (outfile == NULL) {
    printf("can't open file %s to write\n", Q3_OUT_FILE);
  }

  bool first = true;
  for (int i = 0; i < a->size; i++) {
    if (h_out[i] == 0) break;    

    if (first) {
      fprintf(outfile, "%d", h_out[i]);
      first = false;
    } else {
      fprintf(outfile, ",%d", h_out[i]);
    }
  }

  fclose(outfile);

  int_vector_free(a);

  cudaFree(d_in);
  cudaFree(d_ones);
  cudaFree(d_prefix);
  cudaFree(d_out);

  return 0;
}

void int_vector_init(vector *vector) {
  if(vector == NULL)
  {
    return;
  }
  vector -> elements = (int*)malloc(sizeof( int));
  vector -> capacity = 1;
  vector -> size = 0;
}

int int_vector_add(vector* vector, int element) {
  if(vector->size + 1 == vector->capacity)
  {
    int *temp = (int*)realloc(vector->elements, vector->capacity*2 * sizeof (int));
    if(temp == NULL)
    {
      return 0;
    }
    vector -> capacity *= 2;
    vector -> elements = temp;
  }
  vector -> elements[vector->size] = element;
  vector -> size += 1;
  return 1;
}

void int_vector_free(vector *vector){
  free(vector->elements);
  free(vector);
}

//returns offset - difference between size and index of last number and offset
int chopString(char *buf, size_t size){
  int offset = 0;
  int indexOfLastNum = size-2;
  if(isdigit(buf[size-2]))
  {
    int index = size-2;
    while(isdigit(buf[index]) && index > 0)
    {
      index--;
    }
    buf[index] = '\0';
    indexOfLastNum = index+1;
    offset = size - indexOfLastNum -1;//-1 to not copy '\0'
    
  } else 
  {
    offset = 0;
  }    
  return offset;
}

