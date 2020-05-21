#include <cstdio>
#include <cstdlib>
#include <vector>

// Update bucket in parallel.
__global__ void putBucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= n) return;
  atomicAdd(&bucket[key[i]], 1);
}

// Prefix sum for starting indices.
// The starting index is the sum of the number of elements in all
// the buckets with a smaller index.
__global__ void setStartIndex(int *bucket, int *starting_index, int *b, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= range) return;
  for(int j=1; j<range; j<<=1) {
    b[i] = bucket[i] + starting_index[i];
    __syncthreads();
    starting_index[i] += b[i-j];
    __syncthreads();
  }
}

// Initialize ending indices
__global__ void setEndIndex(int *starting_index, int *ending_index, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= range) return;
  ending_index[i] = starting_index[i+1];
}

// Change key value to the corresponding bucket id.
// Since the indices for the keys are non-overlapping,
// we can assign the values in parallel.
__global__ void setKey(int *key, int *starting_index, int *ending_index, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= range) return;
  for(int j = starting_index[i]; j < ending_index[i]; j++) {
    key[j] = i;
  }
}

int main() {
  // M is the number of threads per block.
  const int M = 1024;
  int n = 50;
  int range = 5;
  // Share the key array with the GPU.
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  // Arrays that need to be shared with GPU.
  int *bucket;
  int *starting_index;
  int *ending_index;
  int *b;

  // Initialize arrays
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&starting_index, range*sizeof(int));
  cudaMallocManaged(&ending_index, range*sizeof(int));
  cudaMallocManaged(&b, range*sizeof(int));

  // Perform GPU computations.
  // Use all the threads in the minimum number of blocks needed.
  // This allows us to use the code for larger n and/or range.
  putBucket<<<(n+M-1)/M,M>>>(key, bucket, n);
  cudaDeviceSynchronize();
  setStartIndex<<<(range+M-1)/M,M>>>(bucket, starting_index, b, range);
  cudaDeviceSynchronize();
  setEndIndex<<<(range+M-1)/M,M>>>(starting_index, ending_index, range);
  ending_index[range-1] = n;
  cudaDeviceSynchronize();
  setKey<<<(range+M-1)/M,M>>>(key, starting_index, ending_index, range);
  cudaDeviceSynchronize();

  // Free the space allocated to the arrays.
  cudaFree(key);
  cudaFree(bucket);
  cudaFree(starting_index);
  cudaFree(ending_index);
  // cudaFree(b);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
