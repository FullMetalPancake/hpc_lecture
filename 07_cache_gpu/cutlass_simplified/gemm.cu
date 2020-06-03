#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#define DEBUG

#include <gemm/dispatch.h>
#include <gemm/epilogue_function.h>
#include "util/timer.h"

using namespace cutlass;

float random(float *M, int m, int n) {
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      (&M[0])[i + j * m] = float(drand48());
    }
  }
}

float zeros(float *M, int m, int n) {
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      (&M[0])[i + j * m] = 0.0;
    }
  }
}

void sync_device(float *M, float *d_data, int m, int n)
{
  size_t bytes = m * n * sizeof(float);
  CUDA_PERROR_EXIT(cudaMemcpy(d_data, &M[0], bytes, cudaMemcpyHostToDevice));
}

void sync_host(float *M, float *d_data, int m, int n)
{
    size_t bytes = m * n * sizeof(float);
    CUDA_PERROR_EXIT(cudaMemcpy(&M[0], d_data, bytes, cudaMemcpyDeviceToHost));
}


int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 4096;
  float alpha = 1.0;
  float beta = 0.0;
  static const matrix_transform_t::kind_t TransformA = matrix_transform_t::NonTranspose;
  static const matrix_transform_t::kind_t TransformB = matrix_transform_t::NonTranspose;
  int g_timing_iterations = 10;
  cudaStream_t stream = 0;
  float A[m * k]; random(A, m, k);
  float B[k * n]; random(B, k, n);
  float C[m * n]; zeros(C, m, n);
  float C2[m * n]; zeros(C2, m, n);
  float *d_A; CUDA_PERROR_EXIT(cudaMalloc((void ** )&d_A, sizeof(float) * m * k));
  float *d_B; CUDA_PERROR_EXIT(cudaMalloc((void ** )&d_B, sizeof(float) * k * n));
  float *d_C; CUDA_PERROR_EXIT(cudaMalloc((void ** )&d_C, sizeof(float) * m * n));
  float *d_C2; CUDA_PERROR_EXIT(cudaMalloc((void ** )&d_C2, sizeof(float) * m * n));
  sync_device(A, d_A, m, k);
  sync_device(B, d_B, k, n);
  sync_device(C, d_C, m, n);
  sync_device(C2, d_C2, m, n);
  cublasHandle_t g_cublas_handle;
  cublasCreate(&g_cublas_handle);
  gpu_timer timer;
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    CUDA_PERROR(cublasSgemm(
                            g_cublas_handle,
                            (cublasOperation_t) TransformA,
                            (cublasOperation_t) TransformB,
                            m,
                            n,
                            k,
                            &alpha,
                            &d_A[0],
                            m,
                            &d_B[0],
                            k,
                            &beta,
                            &d_C[0],
                            m));
  }
  timer.stop();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = timer.elapsed_millis() / g_timing_iterations;
  double cublas_flops = double(num_flops) / tcublas / 1.0e6;
  typedef gemm::blas_scaled_epilogue<float, float, float> epilogue_op_t;
  epilogue_op_t epilogue(alpha, beta);
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    gemm::dispatch<epilogue_op_t>(
        m,
        n,
        k,
        alpha,
        beta,
        d_A,
        d_B,
        d_C2,
        stream,
        false);
  }
  timer.stop();
  double tcutlass = timer.elapsed_millis() / g_timing_iterations;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e6;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  sync_host(C, d_C, m, n);
  sync_host(C2, d_C2, m, n);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[i + j * n] - C2[i + j * n]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cublasDestroy(g_cublas_handle);
}