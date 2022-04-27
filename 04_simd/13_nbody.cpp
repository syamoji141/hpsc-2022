#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

float reduce_vector(__m256 vec, float *a) {
  __m256 tmpvec = _mm256_permute2f128_ps(vec,vec,1);
  tmpvec = _mm256_add_ps(tmpvec,vec);
  tmpvec = _mm256_hadd_ps(tmpvec,tmpvec);
  tmpvec = _mm256_hadd_ps(tmpvec,tmpvec);
  _mm256_store_ps(a, tmpvec);
  return a[0];
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], j[N], a[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i] = i;
    a[i] = 0;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 fxvec = _mm256_load_ps(fx);
  __m256 fyvec = _mm256_load_ps(fy);
  __m256 jvec = _mm256_load_ps(j);
  __m256 zerovec = _mm256_set1_ps(0);
  for(int i=0; i<N; i++) {
    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(ivec, jvec, _CMP_NEQ_OQ);
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);
    __m256 rxvec = _mm256_sub_ps(xivec, xvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);
    __m256 rrvec = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec)));
    __m256 rr3vec = _mm256_mul_ps(rrvec, _mm256_mul_ps(rrvec, rrvec));;
    __m256 fxivec = _mm256_blendv_ps(zerovec, _mm256_mul_ps(_mm256_mul_ps(rxvec, mvec), rr3vec), mask);
    __m256 fyivec = _mm256_blendv_ps(zerovec, _mm256_mul_ps(_mm256_mul_ps(ryvec, mvec), rr3vec), mask);
    fx[i] -= reduce_vector(fxivec, a);
    fy[i] -= reduce_vector(fyivec, a);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
