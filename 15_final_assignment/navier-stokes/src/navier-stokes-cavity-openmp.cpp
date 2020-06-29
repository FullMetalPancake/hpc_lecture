//============================================================================
// Name        : navier-stokes-openmp.cpp
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
template<size_t nx, size_t ny>
void save_matrix_to_csv(double (&arr)[nx][ny], string file_name) {
	ofstream file(file_name);
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny - 1; j++) {
			file << arr[i][j] << ',';
		}
		file << arr[i][ny - 1] << '\n';
	}
	file.close();
}

template<size_t nx, size_t ny>
void copy_matrix(double (&from)[nx][ny], double (&to)[nx][ny]) {
#pragma omp parallel for
	for (int i = 0; i < nx; i++) {
#pragma omp parallel for
		for (int j = 0; j < ny; j++) {
			to[i][j] = from[i][j];
		}
	}
}

template<size_t size_x, size_t size_y>
void initialize_matrix(double (&arr)[size_x][size_y]) {
#pragma omp parallel for
	for (int i = 0; i < size_x; i++) {
#pragma omp parallel for
		for (int j = 0; j < size_y; j++) {
			arr[i][j] = 0.0;
		}
	}
}

/*
 * Navier-stokes c++ translation from the provided python code
 */
template<size_t nx, size_t ny>
void build_up_b(double (&b)[nx][ny], double rho, double dt, double (&u)[nx][ny],
		double v[nx][ny], double dx, double dy) {

	for (int i = 1; i < nx - 1; i++) {
#pragma omp parallel for
		for (int j = 1; j < ny - 1; j++) {
			b[i][j] =
					(rho
							* (1 / dt
									* ((u[i][j + 1] - u[i][j - 1]) / (2 * dx)
											+ (v[i + 1][j] - v[i - 1][j])
													/ (2 * dy))
									- ((u[i][j + 1] - u[i][j - 1]) / (2 * dx))
											* ((u[i][j + 1] - u[i][j - 1])
													/ (2 * dx))
									- 2
											* ((u[i + 1][j] - u[i - 1][j])
													/ (2 * dy)
													* (v[i][j + 1] - v[i][j - 1])
													/ (2 * dx))
									- ((v[i + 1][j] - v[i - 1][j]) / (2 * dy))
											* ((v[i + 1][j] - v[i - 1][j])
													/ (2 * dy))));
		}
	}
}

template<size_t nx, size_t ny>
void pressure_poisson(double (&p)[nx][ny], double dx, double dy,
		double (&b)[nx][ny], int nit) {
	double pn[nx][ny];

	for (int i = 0; i < nit; i++) {
		copy_matrix(p, pn);
#pragma omp parallel for
		for (int j = 1; j < nx - 1; j++) {
#pragma omp parallel for
			for (int k = 1; k < ny - 1; k++) {
				p[j][k] = (((pn[j][k + 1] + pn[j][k - 1]) * dy * dy
						+ (pn[j + 1][k] + pn[j - 1][k]) * dx * dx)
						/ (2 * (dx * dx + dy * dy))
						- (dx * dx * dy * dy) / (2 * (dx * dx + dy * dy))
								* b[j][k]);
			}
		}
#pragma omp parallel for
		for (int i = 0; i < nx; i++) {
			p[i][ny - 1] = p[i][ny - 2];
			p[i][0] = p[i][1];
		}

#pragma omp parallel for
		for (int j = 0; j < ny; j++) {
			p[0][j] = p[1][j];
			p[nx - 1][j] = 0;
		}
	}
}

template<size_t nx, size_t ny>
void cavity_flow(int nt, double (&u)[nx][ny], double (&v)[nx][ny], double dt,
		double dx, double dy, double (&p)[nx][ny], double rho, double nu,
		int nit) {
	double un[nx][ny];
	double vn[nx][ny];
	double b[nx][ny];
	initialize_matrix(b);

#pragma omp parallel for
	for (int i = 0; i < nt; i++) {
		copy_matrix(u, un);
		copy_matrix(v, vn);

		build_up_b(b, rho, dt, u, v, dx, dy);
		pressure_poisson(p, dx, dy, b, nit);

#pragma omp parallel for
		for (int j = 1; j < nx - 1; j++) {
#pragma omp parallel for
			for (int k = 1; k < ny - 1; k++) {
				u[j][k] = (un[j][k]
						- un[j][k] * dt / dx * (un[j][k] - un[j][k - 1])
						- vn[j][k] * dt / dy * (un[j][k] - un[j - 1][k])
						- dt / (2 * rho * dx) * (p[j][k + 1] - p[j][k - 1])
						+ nu
								* (dt / (dx * dx)
										* (un[j][k + 1] - 2 * un[j][k]
												+ un[j][k - 1])
										+ dt / (dy * dy)
												* (un[j + 1][k] - 2 * un[j][k]
														+ un[j - 1][k])));

				v[j][k] = (vn[j][k]
						- un[j][k] * dt / dx * (vn[j][k] - vn[j][k - 1])
						- vn[j][k] * dt / dy * (vn[j][k] - vn[j - 1][k])
						- dt / (2 * rho * dy) * (p[j + 1][k] - p[j - 1][k])
						+ nu
								* (dt / (dx * dx)
										* (vn[j][k + 1] - 2 * vn[j][k]
												+ vn[j][k - 1])
										+ dt / (dy * dy)
												* (vn[j + 1][k] - 2 * vn[j][k]
														+ vn[j - 1][k])));
			}
		}
#pragma omp parallel for
		for (int j = 0; j < nx; j++) {
			u[j][0] = 0;
			u[j][ny - 1] = 0;
			v[j][0] = 0;
			v[j][ny - 1] = 0;
		}

#pragma omp parallel for
		for (int j = 0; j < ny; j++) {
			u[0][j] = 0;
			u[nx - 1][j] = 1;
			v[0][j] = 0;
			v[nx - 1][j] = 0;
		}
	}
}

int main() {
	const int nx = 41;
	const int ny = 41;
	const int nt = 500;
	const int nit = 50;
	const double dx = 2.0 / (nx - 1);
	const double dy = 2.0 / (ny - 1);

	const double rho = 1.0;
	const double nu = 0.1;
	const double dt = 0.001;

	double u[nx][ny];
	double v[nx][ny];
	double p[nx][ny];
	double b[nx][ny];

	initialize_matrix(u);
	initialize_matrix(b);
	initialize_matrix(p);
	initialize_matrix(b);

	auto start = std::chrono::high_resolution_clock::now();
	cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, nit);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( stop - start ).count();
	std::cout << duration;

	save_matrix_to_csv(u, "u.csv");
	save_matrix_to_csv(v, "v.csv");
	save_matrix_to_csv(p, "p.csv");
	return 0;

}

