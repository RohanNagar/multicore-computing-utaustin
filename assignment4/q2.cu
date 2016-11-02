#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>

#define INPUT_FILE "inp.txt"
#define NUM_BUCKETS 10
#define Q2A_OUT_FILE "q2a.txt"
#define Q2B_OUT_FILE "q2b.txt"
#define Q2C_OUT_FILE "q2c.txt"

typedef struct vector {
	int *elements;
	int capacity;
	int size;
}vector;

void int_vector_init(vector *vector);
int int_vector_add(vector* vector, int element);
void int_vector_free(vector *vector);

int chopString(char *buf, size_t size);
void bucketize(vector *a, int *b);
int findMin(vector* vector);
void prefixSum(vector *a, int *b);

__global__ void global_reduce_kernel(int * d_out, int * d_in, int * d_intermediate, int size, bool phaseOne)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;
    // do reduction in global mem
	
	if(phaseOne)
	{
		int index = d_in[myId] / 100;
		printf("block dim: %d\n", blockDim.x);
		d_intermediate[myId*NUM_BUCKETS + index] = 1;
		//printf("index: %d value: %d\n", myId*NUM_BUCKETS + index, d_in[myId] );
		__syncthreads(); //every thread computes its bucket for number
	}
	
	//then combine using reduce
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
		
		if(tid < s && myId + s < size)
		{
			for (int i = 0; i < NUM_BUCKETS; i++)
			{
				d_intermediate[myId*NUM_BUCKETS + i] += d_intermediate[(myId + s)* NUM_BUCKETS + i];
			}
			
		}
		__syncthreads();
	}
	
	//copy intermediate to block output
	if (tid == 0)
    {
		for(int i = 0; i < NUM_BUCKETS; i++)
		{
			d_out[blockIdx.x*NUM_BUCKETS + i] = d_intermediate[myId*NUM_BUCKETS + i];
		}
    }
}

__global__ void shmem_reduce_kernel(int * d_out, const int * d_in, int size, bool phaseOne)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ int sdata[];//let's use sdata like we use d_intermediate above
	//__shared__ int b[10] = {0};

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;
	
    // load shared mem from global mem
	
	if(phaseOne)
	{
		int index = d_in[myId] / 100;
		for(int i = 0; i < NUM_BUCKETS; i++)
		{
			if(index == i)
			{
				sdata[tid*NUM_BUCKETS + i] = 1;
			}
			else
			{
				sdata[tid*NUM_BUCKETS + i] = 0;
			}
			
		}
		
	}
	else 
	{
		for(int i = 0; i < NUM_BUCKETS; i++)
		{
			sdata[tid*NUM_BUCKETS + i] = d_in[myId*NUM_BUCKETS + i];
		}
	}
    __syncthreads(); 
	
	           // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s && myId + s < size)
		{
			for (int i = 0; i < NUM_BUCKETS; i++)
			{
				sdata[tid*NUM_BUCKETS + i] += sdata[(tid + s)* NUM_BUCKETS + i];
			}
			
		}
		__syncthreads();
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
		for(int i = 0; i < NUM_BUCKETS; i++)
		{
			d_out[blockIdx.x*NUM_BUCKETS + i] += sdata[tid * NUM_BUCKETS + i];
		}
        
    }
}

__global__ void parallel_prefix_kernel(int * d_out, const int * d_in, int size) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int d = 1;
	d_out[myId] = d_in[myId];
	while(d < size)
	{
		if(myId+1 > d && myId < size)
		{
			d_out[myId] += d_out[myId - d];
		}
		d *= 2;
		__syncthreads();
	}
	
}

void reduce_a(int * b, int * b_intermediate, int * d_intermediate, int * d_in,
            int size)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 1024; //increased from 512 to 1024 to handle 1024^2 values
    int threads = maxThreadsPerBlock;
    int blocks = (size + (maxThreadsPerBlock - 1)) / maxThreadsPerBlock;
	//printf("blocks: %d, size: %d\n", blocks, size);
    global_reduce_kernel<<<blocks, threads>>>(b_intermediate, d_in, d_intermediate, size, true);
	int newSize = blocks;
	int powerOfTwo = 1;
	while (powerOfTwo < blocks) 
	{
		powerOfTwo *= 2;
	}
	threads = powerOfTwo; // launch one thread for each block in prev step
    blocks = 1;

	global_reduce_kernel<<<blocks, threads>>>(b, d_in, b_intermediate, newSize, false);
}

void reduce_b(int * b, int * d_intermediate, int * d_in,
            int size)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 1024; //increased from 512 to 1024 to handle 1024^2 values
    int threads = maxThreadsPerBlock;
    int blocks = (size + (maxThreadsPerBlock - 1)) / maxThreadsPerBlock;
    
    shmem_reduce_kernel<<<blocks, threads, NUM_BUCKETS * threads * sizeof(int)>>>(d_intermediate, d_in, size, true);
    int newSize = blocks;
	int powerOfTwo = 1;
	while (powerOfTwo < blocks) 
	{
		powerOfTwo *= 2;
	}
	threads = powerOfTwo; // launch one thread for each block in prev step
    blocks = 1;

	shmem_reduce_kernel<<<blocks, threads, threads* NUM_BUCKETS * sizeof(int)>>>(b, d_intermediate, newSize, false);

}

void run_c(int *b, int * d_in, int size) {
	parallel_prefix_kernel<<<10, 1>>>(b, d_in, size); 
}

int main(int argc, char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);
	
	cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }

	FILE *fp;
	if((fp = fopen(INPUT_FILE, "r")) == 0)
	{
		printf("%s cannot be found\n", INPUT_FILE);
		exit(-1);
	}
	char separators[] = " ,";
	//int num;
	char buf[100];
	char* token;
	int offset = 0;
	vector *a = (vector*)malloc(sizeof (vector));
	int_vector_init(a);

	while(fgets(buf + offset, sizeof buf - offset, fp) != NULL)
	{
		//chop off number from string if it ends with digit
		offset = chopString(buf, sizeof buf);	
		int indexOfLastNum = sizeof buf - offset - 1;// -1 to not copy '\0'

		//printf("buffer: %s\n", buf);	
	
		token = strtok(buf, separators);
		while (token != NULL)
		{
			int num = atoi(token);
			//printf("%d\n", num); 
			int_vector_add(a, num);
			token = strtok(NULL, separators);
		}
		memcpy(buf, &buf[indexOfLastNum], offset);
		
			
	}
	int min = findMin(a); 
	int b[10] = {0};
	bucketize(a, b);
	for(int i = 0; i < NUM_BUCKETS; i++)
	{
		printf("index: %d count: %d\n", i, b[i]);
	}
	
	const int ARRAY_BYTES = sizeof(int) * a->size;

    // declare GPU memory pointers
    int * d_in, * d_intermediate, * b_intermediate, *a_out, *b_out, *c_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES * NUM_BUCKETS); // overallocated
	cudaMalloc((void **) &b_intermediate, ARRAY_BYTES * NUM_BUCKETS);
    //cudaMalloc((void **) &d_out, sizeof(int));
	cudaMalloc((void **) &a_out, sizeof(int) * NUM_BUCKETS);
	cudaMalloc((void **) &b_out, sizeof(int) * NUM_BUCKETS);
	cudaMalloc((void **) &c_out, sizeof(int) * NUM_BUCKETS);
	//free CPU memory
	
	
	//cudaMemset(b_out, 0, sizeof(int) * NUM_BUCKETS);
	cudaMemset(b_out, 0, sizeof(int) * NUM_BUCKETS);
	cudaMemset(a_out, 0, sizeof(int) * NUM_BUCKETS);
	
	cudaMemset(b_intermediate, 0, ARRAY_BYTES * NUM_BUCKETS);//
	cudaMemset(d_intermediate, 0, ARRAY_BYTES * NUM_BUCKETS);//
    // transfer the input array to the GPU
    cudaMemcpy(d_in, a->elements, ARRAY_BYTES, cudaMemcpyHostToDevice);
	
	int_vector_free(a);

	
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
    // launch the kernel
	reduce_a(a_out, b_intermediate,  d_intermediate, d_in, a->size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventRecord(start, 0);
	reduce_b(b_out, d_intermediate, d_in, a->size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventRecord(start, 0);
	run_c(c_out, b_out, NUM_BUCKETS);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
	int h_a_out[NUM_BUCKETS];
	int h_b_out[NUM_BUCKETS];
	int h_c_out[NUM_BUCKETS];
	
	// copy back data from GPU
    cudaMemcpy(h_a_out, a_out, sizeof(int) * NUM_BUCKETS, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b_out, b_out, sizeof(int) * NUM_BUCKETS, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c_out, c_out, sizeof(int) * NUM_BUCKETS, cudaMemcpyDeviceToHost);
	
	/****************** write to q2a.txt *******************************/
	
	FILE *outFp = fopen(Q2A_OUT_FILE, "w");
	if(outFp == NULL)
	{
		printf("can't open file %s to write\n", Q2A_OUT_FILE);
	}
	fprintf(outFp, "%d", h_a_out[0]);
	for(int i = 1; i < NUM_BUCKETS; i++)
	{
		fprintf(outFp, ", %d", h_a_out[i]);
	}
	fprintf(outFp, "\n");
	
	fclose(outFp);
	
	/*********************** write to q2b.txt ***********************/
	
	outFp = fopen(Q2B_OUT_FILE, "w");
	if(outFp == NULL)
	{
		printf("can't open file %s to write\n", Q2A_OUT_FILE);
	}
	fprintf(outFp, "%d", h_b_out[0]);
	for(int i = 1; i < NUM_BUCKETS; i++)
	{
		fprintf(outFp, ", %d", h_b_out[i]);
	}
	fprintf(outFp, "\n");
	
	fclose(outFp);
	
	/***********************debug parallel prefix sum*********************/
	printf("parallel prefix sum cpu: ");
	vector * wrapper = (vector*) malloc(sizeof(vector));
	int_vector_init(wrapper);
	for (int i = 0; i < NUM_BUCKETS; i++)
	{
		int_vector_add(wrapper, h_b_out[i]);
	}
	int cpu_prefix_out[10] = {0};
	prefixSum(wrapper, cpu_prefix_out);
	printf("%d", cpu_prefix_out[0]);
	for(int i = 1; i < NUM_BUCKETS; i++)
	{
		printf( ", %d", cpu_prefix_out[i]);
	}
	printf("\n");
	
	/************************write to q2c.txt***********************************/
	
	
	outFp = fopen(Q2C_OUT_FILE, "w");
	if(outFp == NULL)
	{
		printf("can't open file %s to write\n", Q2C_OUT_FILE);
	}
	fprintf(outFp, "%d", h_c_out[0]);
	for(int i = 1; i < NUM_BUCKETS; i++)
	{
		fprintf(outFp, ", %d", h_c_out[i]);
	}
	fprintf(outFp, "\n");
	
	fclose(outFp);
	
	
    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
	cudaFree(b_intermediate);
	cudaFree(a_out);
	cudaFree(b_out);
	cudaFree(c_out);
    

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

//input: a array of int; output: b array of 10 int
void bucketize(vector *a, int *b) {
	for(int i = 0; i < a->size; i++)
	{
		int index = a->elements[i] / 100;
		b[index] += 1;
	}
}

//goal is a running sum of a
//b must have 10 elements
void prefixSum(vector *a, int *b) {
	int * elements = a->elements;
	vector * firstPass = (vector*) malloc(sizeof(vector));
	int_vector_init(firstPass);
	int size = a->size;
	b[0] = elements[0];
	for(int i = 1; i < size; i++)
	{
		//int_vector_add(firstPass, sum);
		b[i] = b[i-1] + elements[i];
	}
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
