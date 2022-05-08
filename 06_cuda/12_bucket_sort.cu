#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void initialize(int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bucket[i] = 0;
}

__global__ void count(int *key,int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void sort(int *key, int *bucket, int n) {
  int k = 0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int j=0; j<i; j++) {
    k += bucket[j];
  }
  for (; bucket[i]>0; bucket[i]--) {
      key[k++] = i;
  }
}


int main() {
  int n = 50;
  int range = 5;
  int *key;
  int *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  initialize<<<1,1024>>>(bucket, range);
  cudaDeviceSynchronize();
  count<<<1,1024>>>(key, bucket, n);
  cudaDeviceSynchronize();
  sort<<<1,1024>>>(key, bucket, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);

}
