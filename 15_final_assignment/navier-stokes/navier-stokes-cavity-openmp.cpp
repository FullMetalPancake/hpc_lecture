//============================================================================
// Name        : navier-stokes-cavity-openmp.cpp
// Author      : Steven Ge
// Description : navier-stokes implementation in c++.
// The code is an extension of the c++ code. It implements
// OpenMP, which allows for parallelization.
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
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		to[i] = from[i];
	}
}

template<size_t n>
void initialize_matrix(double (&arr)[n]) {
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		arr[i] = 0.0;
	}
}

/*
 * Navier-stokes c++ translation from the provided python code
 */
template<size_t n>
void build_up_b(double (&b)[n], double rho, double dt, double (&u)[n],
		double v[n], double dx, double dy, int nx, int ny) {

	for (int i = 1; i < nx - 1; i++) {
#pragma omp parallel for
		for (int j = 1; j < ny - 1; j++) {
			b[i*nx + j] =
			(rho
					* (1 / dt
							* ((u[i*nx + j + 1] - u[i*nx + j - 1]) / (2 * dx)
									+ (v[(i + 1)*nx + j] - v[(i - 1)*nx + j])
									/ (2 * dy))
							- ((u[i*nx + j + 1] - u[i*nx + j - 1]) / (2 * dx))
							* ((u[i*nx + j + 1] - u[i*nx + j - 1])
									/ (2 * dx))
							- 2
							* ((u[(i + 1)*nx + j] - u[(i - 1)*nx + j])
									/ (2 * dy)
									* (v[i*nx + j + 1] - v[i*nx + j - 1])
									/ (2 * dx))
							- ((v[(i + 1)*nx + j] - v[(i - 1)*nx + j]) / (2 * dy))
							* ((v[(i + 1)*nx + j] - v[(i - 1)*nx + j])
									/ (2 * dy))));
		}
	}
}

template<size_t n>
void pressure_poisson(double (&p)[n], double dx, double dy,
		double (&b)[n], int nit, int nx, int ny) {
	double pn[n];

	for (int i = 0; i < nit; i++) {
		copy_matrix(p, pn);
#pragma omp parallel for
		for (int j = 1; j < nx - 1; j++) {
#pragma omp parallel for
			for (int k = 1; k < ny - 1; k++) {
				p[j*nx + k] = (((pn[j*nx + k + 1] + pn[j*nx + k - 1]) * dy * dy
								+ (pn[(j + 1)*nx + k] + pn[(j - 1)*nx + k]) * dx * dx)
						/ (2 * (dx * dx + dy * dy))
						- (dx * dx * dy * dy) / (2 * (dx * dx + dy * dy))
						* b[j*nx + k]);
			}
		}
#pragma omp parallel for
		for (int j = 0; j < nx; j++) {
			p[j*nx + ny - 1] = p[j*nx + ny - 2];
			p[j*nx] = p[j*nx + 1];
		}

#pragma omp parallel for
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

#pragma omp parallel for
	for (int i = 0; i < nt; i++) {
		copy_matrix(u, un);
		copy_matrix(v, vn);

		build_up_b(b, rho, dt, u, v, dx, dy, nx, ny);
		pressure_poisson(p, dx, dy, b, nit, nx, ny);

#pragma omp parallel for
		for (int j = 1; j < nx - 1; j++) {
#pragma omp parallel for
			for (int k = 1; k < ny - 1; k++) {
				u[j*nx + k] = (un[j*nx + k]
						- un[j*nx + k] * dt / dx * (un[j*nx + k] - un[j*nx + k - 1])
						- vn[j*nx + k] * dt / dy * (un[j*nx + k] - un[(j - 1)*nx + k])
						- dt / (2 * rho * dx) * (p[j*nx + k + 1] - p[j*nx + k - 1])
						+ nu
						* (dt / (dx * dx)
								* (un[j*nx + k + 1] - 2 * un[j*nx + k]
										+ un[j*nx + k - 1])
								+ dt / (dy * dy)
								* (un[(j + 1)*nx + k] - 2 * un[j*nx + k]
										+ un[(j - 1)*nx + k])));

				v[j*nx + k] = (vn[j*nx + k]
						- un[j*nx + k] * dt / dx * (vn[j*nx + k] - vn[j*nx + k - 1])
						- vn[j*nx + k] * dt / dy * (vn[j*nx + k] - vn[(j - 1)*nx + k])
						- dt / (2 * rho * dy) * (p[(j + 1)*nx + k] - p[(j - 1)*nx + k])
						+ nu
						* (dt / (dx * dx)
								* (vn[j*nx + k + 1] - 2 * vn[j*nx + k]
										+ vn[j*nx + k - 1])
								+ dt / (dy * dy)
								* (vn[(j + 1)*nx + k] - 2 * vn[j*nx + k]
										+ vn[(j - 1)*nx + k])));
			}
		}
#pragma omp parallel for
		for (int j = 0; j < nx; j++) {
			u[j*nx] = 0;
			u[j*nx + ny - 1] = 0;
			v[j*nx] = 0;
			v[j*nx + ny - 1] = 0;
		}

#pragma omp parallel for
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

	double u[nx * ny];
	double v[nx * ny];
	double p[nx * ny];
	double b[nx * ny];

	initialize_matrix(u);
	initialize_matrix(b);
	initialize_matrix(p);
	initialize_matrix(b);

	auto start = std
	::chrono::high_resolution_clock::now();
	cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, nit, nx, ny);
	auto stop = std
	::chrono::high_resolution_clock::now();
	auto duration = std
	::chrono::duration_cast<std::chrono::seconds>( stop - start ).count();
	ofstream
	file("result/milliseconds-openmp");
	file << duration;
	file.close();

	save_matrix_to_csv(u, "result/u-openmp.csv", nx, ny);
	save_matrix_to_csv(v, "result/v-openmp.csv", nx, ny);
	save_matrix_to_csv(p, "result/p-openmp.csv", nx, ny);
	return 0;

}

