#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], rx[N][N], ry[N][N], r[N][N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    // printf("%d %g %g %g\n",i,x[i],y[i],m[i]);
  }
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
      if(i != j) {
        rx[i][j] = x[i] - x[j];
        ry[i][j] = y[i] - y[j];
        r[i][j] = std::sqrt(rx[i][j] * rx[i][j] + ry[i][j] * ry[i][j]);
        fx[i] -= rx[i][j] * m[j] / (r[i][j] * r[i][j] * r[i][j]);
        fy[i] -= ry[i][j] * m[j] / (r[i][j] * r[i][j] * r[i][j]);
      }
      // printf("%d %g\n",i,r[i][j]);
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
