#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void putBucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= n) return;
  atomicAdd(&bucket[key[i]], 1);
}

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

__global__ void setEndIndex(int *starting_index, int *ending_index, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= range) return;
  ending_index[i] = starting_index[i+1];
}

__global__ void setKey(int *key, int *starting_index, int *ending_index, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= range) return;
  for(int j = starting_index[i]; j < ending_index[i]; j++) {
    key[j] = i;
  }
}

int main() {
  const int M = 1024;
  int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  int *starting_index;
  int *ending_index;
  int *b;

  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&starting_index, range*sizeof(int));
  cudaMallocManaged(&ending_index, range*sizeof(int));
  cudaMallocManaged(&b, range*sizeof(int));

  putBucket<<<(n+M-1)/M,M>>>(key, bucket, n);
  cudaDeviceSynchronize();
  setStartIndex<<<(range+M-1)/M,M>>>(bucket, starting_index, b, range);
  cudaDeviceSynchronize();
  setEndIndex<<<(range+M-1)/M,M>>>(starting_index, ending_index, range);
  ending_index[range-1] = n;
  cudaDeviceSynchronize();
  setKey<<<(range+M-1)/M,M>>>(key, starting_index, ending_index, range);
  cudaDeviceSynchronize();

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
