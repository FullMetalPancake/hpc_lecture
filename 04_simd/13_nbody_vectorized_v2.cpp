#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

float vector_reduction_add(__m256 avec, __m256 f) {
  __m256 bvec = _mm256_permute2f128_ps(avec,avec,1);
  bvec = _mm256_add_ps(bvec,avec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  return bvec[0];
}

int main() {
  const int N = 8;
  // Create additional stack arrays for reduction later
  float x[N], y[N], m[N], fx[N], fy[N], fx_store[N], fy_store[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  // Load x and y vectors to registers.
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);

  // Initialize 0 vector and 1 vector for creating the masks.
  __m256 zero = _mm256_set1_ps(0);
  __m256 one = _mm256_set1_ps(1);

  for(int i=0; i<N; i++) {
    // Load m vector of bodies.
    // Reload every iteration, as the ith entry needs to be set to 0.
    __m256 mvec = _mm256_load_ps(m);

    // Create vector with only x[i] and y[i].
    // Needed for subtraction.
    __m256 x_i = _mm256_set1_ps(x[i]);
    __m256 y_i = _mm256_set1_ps(y[i]);

    // Compute r using the formula.
    __m256 rx = _mm256_sub_ps(x_i, xvec);
    __m256 ry = _mm256_sub_ps(y_i, yvec);
    __m256 rx_square = _mm256_mul_ps(rx, rx);
    __m256 ry_square = _mm256_mul_ps(ry, ry);
    __m256 r_add = _mm256_add_ps(rx_square, ry_square);
    __m256 r = _mm256_sqrt_ps(r_add);

    // Set ith entry of the m vector to 0.
    __m256 r_mask = _mm256_cmp_ps(r, zero, _CMP_GT_OQ);
    mvec = _mm256_blendv_ps(zero, mvec, r_mask);

    // Set the ith entry of the r_cubed vector to 1.
    // This prevents division by zero. Since the ith entry of the m vector is
    // set to 0, we get the same result as the (i != j) if-statement.
    __m256 r_cubed = _mm256_mul_ps(r, r_add);
    r_cubed = _mm256_blendv_ps(one, r_cubed, r_mask);

    // Do the final calculations to get the vector to be reduced.
    __m256 fxvec = _mm256_mul_ps(rx, mvec);
    __m256 fyvec = _mm256_mul_ps(ry, mvec);
    fxvec = _mm256_div_ps(fxvec, r_cubed);
    fyvec = _mm256_div_ps(fyvec, r_cubed);

    // Intrinsic (Vector Reduction) for fxvec.
    // Store the result of the reduction in fx.
    __m256 bvec = _mm256_permute2f128_ps(fxvec,fxvec,1);
    bvec = _mm256_add_ps(bvec,fxvec);
    bvec = _mm256_hadd_ps(bvec,bvec);
    bvec = _mm256_hadd_ps(bvec,bvec);
    _mm256_store_ps(fx_store, bvec);
    fx[i] = fx_store[0];

    // Intrinsic (Vector Reduction) for fyvec.
    // Store the result of the reduction in fy.
    bvec = _mm256_permute2f128_ps(fyvec,fyvec,1);
    bvec = _mm256_add_ps(bvec,fyvec);
    bvec = _mm256_hadd_ps(bvec,bvec);
    bvec = _mm256_hadd_ps(bvec,bvec);
    _mm256_store_ps(fy_store, bvec);
    fy[i] = fy_store[0];
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
