#include <cstdio>
#include <openacc.h>

int main() {
  int N = 8;
  int a = 0;
  int b[N];
#pragma acc parallel loop private(a)
  for(int i=0; i<N; i++) {
    // a = __pgi_vectoridx();
    b[i] = __pgi_vectoridx();
  }
  printf("%d",a);
  for(int i=0; i<N; i++) {
    printf(" %d",b[i]);
  }
  printf("\n");
}
