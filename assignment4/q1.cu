#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>

#define INPUT_FILE "inp.txt"
#define Q1A_OUT_FILE "q1a.txt"
#define Q1B_OUT_FILE "q1b.txt"

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

int findMin(vector* vector);

// PART A
__global__ void global_reduce_kernel(int * d_out, int * d_in, int size) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid  = threadIdx.x;

  // do reduction in global mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && myId + s < size) {
      // d_in[myId] += d_in[myId + s];
      if(d_in[myId+s] < d_in[myId]) {
        d_in[myId] = d_in[myId+s];
      }
    }

    __syncthreads();        // make sure all moves at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0) {
    d_out[blockIdx.x] = d_in[myId];
  }
}

void reduce(int * d_out, int * d_intermediate, int * d_in,
            int size, bool usesSharedMemory) {
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 1024; //increased from 512 to 1024 to handle 1024^2 values
    int threads = maxThreadsPerBlock;
    int blocks = (size + (maxThreadsPerBlock-1)) / maxThreadsPerBlock;
    global_reduce_kernel<<<blocks, threads>>>
        (d_intermediate, d_in, size);

    int powerOfTwo = 1;
    while (powerOfTwo < blocks) {
      powerOfTwo *= 2;
    }

    // now we're down to one block left, so reduce it
    int newSize = blocks;
    threads = powerOfTwo; // launch one thread for each block in prev step
    blocks = 1;
    global_reduce_kernel<<<blocks, threads>>>
        (d_out, d_intermediate, newSize);
}

// PART B
__global__ void partb(int* d_out, int* d_in, int size) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;

  if (myId < size) {
    int f = d_in[myId];
    d_out[myId] = f % 10;
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

  FILE *fp;
  if((fp = fopen(INPUT_FILE, "r")) == 0) {
    printf("%s cannot be found\n", INPUT_FILE);
    exit(-1);
  }
  char separators[] = " ,";
  char buf[100];
  char* token;
  int offset = 0;
  vector *a = (vector*) malloc(sizeof (vector));
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
  int min = findMin(a); 
  const int ARRAY_BYTES = sizeof(int) * a->size;

  // declare GPU memory pointers
  int * d_in, * d_in_b, * d_intermediate, * d_out, * d_out_b;

  // allocate GPU memory
  cudaMalloc((void **) &d_in_b, ARRAY_BYTES);
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
  cudaMalloc((void **) &d_out_b, ARRAY_BYTES);
  cudaMalloc((void **) &d_out, sizeof(int));

  // transfer the input array to the GPU
  cudaMemcpy(d_in, a->elements, ARRAY_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_b, a->elements, ARRAY_BYTES, cudaMemcpyHostToDevice);
  int whichKernel = 0;
  if (argc == 2) {
      whichKernel = atoi(argv[1]);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // launch the kernel
  switch(whichKernel) {
    case 0:
      //printf("Running global reduce\n");
      cudaEventRecord(start, 0);
      reduce(d_out, d_intermediate, d_in, a->size, false);
      cudaEventRecord(stop, 0);
      break;
    case 1:
      //printf("Running reduce with shared mem\n");
      cudaEventRecord(start, 0);
      reduce(d_out, d_intermediate, d_in, a->size, true);
      cudaEventRecord(stop, 0);
      break;
    default:
      fprintf(stderr, "error: ran no kernel\n");
      exit(EXIT_FAILURE);
  }
  
  // PART B 
  const int maxThreadsPerBlock = 1024; //increased from 512 to 1024 to handle 1024^2 values
  int threads = maxThreadsPerBlock;
  int blocks = (a->size + (maxThreadsPerBlock-1)) / maxThreadsPerBlock;
  partb<<<blocks, threads>>>(d_out_b, d_in_b, a->size);
  
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // copy back the sum from GPU
  int h_out;
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

  int h_out_b[a->size];
  cudaMemcpy(h_out_b, d_out_b, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  //printf("average time elapsed: %f\n", elapsedTime);
  FILE *outFp = fopen(Q1A_OUT_FILE, "w");
  if(outFp == NULL) {
    printf("can't open file %s to write\n", Q1A_OUT_FILE);
  }
  
  fprintf(outFp, "%d\n", h_out);
  fclose(outFp);
  
  FILE *outFpB = fopen(Q1B_OUT_FILE, "w");
  if (outFpB == NULL) {
    printf("can't open file %s to write\n", Q1B_OUT_FILE);
  }

  bool first = true;
  for (int i = 0; i < a->size; i++) {
    if (first) {
      fprintf(outFpB, "%d", h_out_b[i]);
      first = false;
    } else {
      fprintf(outFpB, ",%d", h_out_b[i]);
    }
  }

  fclose(outFpB);

  int_vector_free(a);
  // free GPU memory allocation
  cudaFree(d_in);
  cudaFree(d_intermediate);
  cudaFree(d_out);
  cudaFree(d_out_b);

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

//assumes vector size >= 1
int findMin(vector* vector) {
  int min = INT_MAX;
  if(vector == NULL)
  {
    return min;
  }
  int size = vector->size;
  int* arr = vector->elements;
  
  for(int i = 0; i < size; i++)
  {
    if(arr[i] < min)
    {
      min = arr[i];
    }
  }
  return min;
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

