//============================================================================
// Name        : navier-stokes-cavity.cpp
// Author      : Steven Ge
// Description : navier-stokes implementation in c++.
//				 The code is a translation of the provided python code.
//============================================================================

#include <iostream>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std;

/*
 * Helper functions
 */
template<size_t n>
void save_matrix_to_csv(double (&arr)[n], string file_name, int nx, int ny) {
	ofstream file(file_name);
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny - 1; j++) {
            file << arr[i*nx + j] << ',';
		}
        file << arr[i*nx + ny - 1] << '\n';
	}
	file.close();
}

template<size_t n>
void copy_matrix(double (&from)[n], double (&to)[n]) {
	for (int i = 0; i < n; i++) {
			to[i] = from[i];
	}
}

template<size_t n>
void initialize_matrix(double (&arr)[n]) {
	for (int i = 0; i < n; i++) {
        arr[i] = 0.0;
	}
}

/*
 * Navier-stokes c++ translation from the provided python code
 */
template<size_t n>
void build_up_b(double (&b)[n], double rho, double dt, double (&u)[n],
		double (&v)[n], double dx, double dy, int nx, int ny, int temp) {
    
    

	for (int k = 0; k < (nx - 2)*(ny - 2); k++) {
    int i = k / (ny-2);
    int j = k % (ny-2);
			b[(i+1)*nx + j + 1] =
					(rho
							* (1 / dt
									* ((u[(i+1)*nx + j + 2] - u[(i+1)*nx + j]) / (2 * dx)
											+ (v[(i + 2)*nx + j + 1] - v[i*nx + j + 1])
													/ (2 * dy))
									- ((u[(i+1)*nx + j + 2] - u[(i+1)*nx + j]) / (2 * dx))
											* ((u[(i+1)*nx + j + 2] - u[(i+1)*nx + j])
													/ (2 * dx))
									- 2
											* ((u[(i + 2)*nx + j + 1] - u[i*nx + j + 1])
													/ (2 * dy)
													* (v[(i+1)*nx + j + 2] - v[(i+1)*nx + j])
													/ (2 * dx))
									- ((v[(i + 2)*nx + j + 1] - v[i*nx + j + 1]) / (2 * dy))
											* ((v[(i + 2)*nx + j + 1] - v[i*nx + j + 1])
													/ (2 * dy))));
	}
	
	save_matrix_to_csv(b, "result/test/b.csv" + std::to_string(temp), nx, ny);
}

template<size_t n>
void pressure_poisson(double (&p)[n], double dx, double dy,
		double (&b)[n], int nit, int nx, int ny) {
	double pn[n];

	for (int i = 0; i < nit; i++) {
		copy_matrix(p, pn);
        
		for (int j = 0; j < (nx - 2)*(ny - 2); j++) {
            int k = j / (ny - 2);
            int l = j % (ny - 2);
            
            p[(k+1)*nx + l+1] = (((pn[(k+1)*nx + l + 2] + pn[(k+1)*nx + l]) * dy * dy
                    + (pn[(k + 2)*nx + l + 1] + pn[k*nx + l + 1]) * dx * dx)
                    / (2 * (dx * dx + dy * dy))
                    - (dx * dx * dy * dy) / (2 * (dx * dx + dy * dy))
                            * b[(k+1)*nx + l + 1]);
        }
		
		for (int j = 0; j < nx; j++) {
			p[j*nx + ny - 1] = p[j*nx + ny - 2];
			p[j*nx] = p[j*nx + 1];
		}

		for (int j = 0; j < ny; j++) {
			p[j] = p[nx+ j];
			p[(nx - 1)*nx + j] = 0;
		}
	}
}

template<size_t n>
void cavity_flow(int nt, double (&u)[n], double (&v)[n], double dt,
		double dx, double dy, double (&p)[n], double rho, double nu,
		int nit, int nx, int ny) {
	double un[n];
	double vn[n];
	double b[n];
	initialize_matrix(b);

	for (int i = 0; i < nt; i++) {
		copy_matrix(u, un);
		copy_matrix(v, vn);

		build_up_b(b, rho, dt, u, v, dx, dy, nx, ny, i);
		pressure_poisson(p, dx, dy, b, nit, nx, ny);
        save_matrix_to_csv(p, "result/test/p" + std::to_string(i), nx, ny);

		for (int j = 0; j < (nx - 2)*(ny - 2); j++) {
            int k = j / (ny - 2);
            int l = j % (ny - 2);
            u[(k+1)*nx + l+1] = (un[(k+1)*nx + l+1]
                    - un[(k+1)*nx + l+1] * dt / dx * (un[(k+1)*nx + l+1] - un[(k+1)*nx + l])
                    - vn[(k+1)*nx + l+1] * dt / dy * (un[(k+1)*nx + l+1] - un[k*nx + l+1])
                    - dt / (2 * rho * dx) * (p[(k+1)*nx + l + 2] - p[(k+1)*nx + l])
                    + nu
                            * (dt / (dx * dx)
                                    * (un[(k+1)*nx + l + 2] - 2 * un[(k+1)*nx + l + 1]
                                            + un[(k+1)*nx + l])
                                    + dt / (dy * dy)
                                            * (un[(k + 2)*nx + l + 1] - 2 * un[(k+1)*nx + l + 1]
                                                    + un[k*nx + l + 1])));

            v[(k+1)*nx + l + 1] = (vn[(k+1)*nx + l + 1]
                    - un[(k+1)*nx + l+1] * dt / dx * (vn[(k+1)*nx + l+1] - vn[(k+1)*nx + l])
                    - vn[(k+1)*nx + l+1] * dt / dy * (vn[(k+1)*nx + l+1] - vn[k*nx + l+1])
                    - dt / (2 * rho * dy) * (p[(k + 2)*nx + l+1] - p[k*nx + l+1])
                    + nu
                            * (dt / (dx * dx)
                                    * (vn[(k+1)*nx + l + 2] - 2 * vn[(k+1)*nx + l+1]
                                            + vn[(k+1)*nx + l])
                                    + dt / (dy * dy)
                                            * (vn[(k + 2)*nx + l+1] - 2 * vn[(k+1)*nx + l+1]
                                                    + vn[k*nx + l+1])));
        }
		
		for (int j = 0; j < nx; j++) {
			u[j*nx] = 0;
			u[j*nx + ny - 1] = 0;
			v[j*nx] = 0;
			v[j*nx + ny - 1] = 0;
		}

		for (int j = 0; j < ny; j++) {
			u[j] = 0;
			u[(nx - 1)*nx + j] = 1;
			v[j] = 0;
			v[(nx - 1)*nx + j] = 0;
		}
	}
}

int main() {
	const int nx = 100;
	const int ny = 100;
	const int nt = 500;
	const int nit = 50;
	const double dx = 2.0 / (nx - 1);
	const double dy = 2.0 / (ny - 1);

	const double rho = 1.0;
	const double nu = 0.1;
	const double dt = 0.001;

	double u[nx*ny];
	double v[nx*ny];
	double p[nx*ny];
	double b[nx*ny];

	initialize_matrix(u);
	initialize_matrix(b);
	initialize_matrix(p);
	initialize_matrix(b);

	auto start = std::chrono::high_resolution_clock::now();
	cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, nit, nx, ny);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( stop - start ).count();
    ofstream file("result/milliseconds");
	file << duration;
	file.close();

	save_matrix_to_csv(u, "result/u.csv", nx, ny);
	save_matrix_to_csv(v, "result/v.csv", nx, ny);
	save_matrix_to_csv(p, "result/p.csv", nx, ny);
	return 0;

}

