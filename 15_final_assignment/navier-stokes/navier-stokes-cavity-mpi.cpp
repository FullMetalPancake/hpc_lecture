//============================================================================
// Name        : navier-stokes-mpi.cpp
// Author      : Steven Ge
// Description : navier-stokes implementation in c++.
//				 The code is a translation of the provided python code.
//============================================================================

#include <iostream>
#include <iostream>
#include <fstream>
#include <chrono>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>     // std::string, std::to_string
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

void save_matrix_to_csv(double *arr, string file_name, int nx, int ny) {
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

template<size_t n>
void vector_addition(double (&arr1)[n], double (&arr2)[n], double (&result)[n]) {
	for (int i = 0; i < n; i++) {
        result[i] = arr1[i] + arr2[i];
    }
}

void initialize_matrix(double *arr, int n) {
	for (int i = 0; i < n; i++) {
        arr[i] = 0.0;
	}
}

template<size_t n>
void get_sub_vector(double *source, double (&target)[n], int begin, int end) {
	for (int i = begin; i < end; i++) {
        target[i-begin] = source[i];
    }
}

void get_sub_vector(double *source, double *target, int begin, int end) {
	for (int i = begin; i < end; i++) {
        target[i-begin] = source[i];
    }
}

void broadcast_parser(double *bcast_data, double *local_data, int nx, int ny, int N) {
    
    for (int k = 0; k < N; k++) {
        int i = k / (ny-2);
        int j = k % (ny-2);
        local_data[(i+1)*nx + j + 1] = bcast_data[k];
	}
}

void allgatherv_helper(int size, int rank, int N, double *bcast, double *data, int nx, int ny) {
    /* break up the elements */
    int *counts = new int[size];
    int *disps  = new int[size];

    for (int i=0; i<size; i++) {
        counts[i] = N / size;
    }

    disps[0] = 0;
    for (int i=1; i<size; i++) {
        disps[i] = disps[i-1] + counts[i-1];
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                    &(bcast[0]), counts, disps, MPI_INT, MPI_COMM_WORLD);

    broadcast_parser(bcast, data, nx, ny, N);
    
    delete [] disps;
    delete [] counts;
}


/*
 * Navier-stokes c++ translation from the provided python code
 */
template<size_t n>
void build_up_b(double (&b)[n], double rho, double dt, double (&u)[n],
		double (&v)[n], double dx, double dy, int nx, int ny, int size, int rank, int temp) {
    
    int N = (nx - 2)*(ny - 2);
    
	int begin = (N / size) * rank;
	int end = (N / size) * (rank + 1);
    
    double b_bcast[N];
    
	for (int k = begin; k < end; k++) {
        int i = k / (ny-2);
        int j = k % (ny - 2);
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
        
        b_bcast[k] = b[(i+1)*nx + j + 1];
	}
    
    MPI_Barrier(MPI_COMM_WORLD);
    allgatherv_helper(size, rank, N, b_bcast, b, nx, ny);
    save_matrix_to_csv(b, "result/test/b-mpi.csv" + std::to_string(temp), nx, ny);
}


template<size_t n>
void pressure_poisson(double (&p)[n], double dx, double dy,
		double (&b)[n], int nit, int nx, int ny, int size, int rank) {
	double pn[n];
    
    int N = (nx - 2)*(ny - 2);
    
    int begin = (N / size) * rank;
	int end = (N / size) * (rank + 1);
    double p_bcast[N];
    
	for (int i = 0; i < nit; i++) {
		copy_matrix(p, pn);
        
		for (int j = begin; j < end; j++) {
            int k = j / (ny - 2);
            int l = j % (ny - 2);
            p[(k+1)*nx + l+1] = (((pn[(k+1)*nx + l + 2] + pn[(k+1)*nx + l]) * dy * dy
                    + (pn[(k + 2)*nx + l + 1] + pn[k*nx + l + 1]) * dx * dx)
                    / (2 * (dx * dx + dy * dy))
                    - (dx * dx * dy * dy) / (2 * (dx * dx + dy * dy))
                            * b[(k+1)*nx + l + 1]);
            
            p_bcast[j] = p[(k+1)*nx + l+1];
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
	
	MPI_Barrier(MPI_COMM_WORLD);
    allgatherv_helper(size, rank, N, p_bcast, p, nx, ny);
}


template<size_t n>
void cavity_flow(int nt, double (&u)[n], double (&v)[n], double dt,
		double dx, double dy, double (&p)[n], double rho, double nu,
		int nit, int nx, int ny, int size, int rank) {
    
    int N = (nx - 2)*(ny - 2);
    
    int begin = (N / size) * rank;
	int end = (N / size) * (rank + 1);
    
	double un[n];
	double vn[n];
	double b[n];
	initialize_matrix(b);
    
    double u_bcast[N];
    double v_bcast[N];
    

	for (int i = 0; i < nt; i++) {
		copy_matrix(u, un);
		copy_matrix(v, vn);

		build_up_b(b, rho, dt, u, v, dx, dy, nx, ny, size, rank, i);
		pressure_poisson(p, dx, dy, b, nit, nx, ny, size, rank);

		for (int j = begin; j < end; j++) {
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
        
            u_bcast[j] = u[(k+1)*nx + l+1];
            v_bcast[j] = v[(k+1)*nx + l + 1];
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
	
	
	MPI_Barrier(MPI_COMM_WORLD);
    allgatherv_helper(size, rank, N, u_bcast, u, nx, ny);
    allgatherv_helper(size, rank, N, v_bcast, v, nx, ny);
}


int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
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

	initialize_matrix(u);
	initialize_matrix(v);
	initialize_matrix(p);

	auto start = std::chrono::high_resolution_clock::now();
	cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, nit, nx, ny, size, rank);


	if(rank == size - 1) {
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>( stop - start ).count();
		ofstream file("result/milliseconds-mpi-" + std::to_string(size));
		file << duration;
		file.close();

		save_matrix_to_csv(u, "result/u-mpi-" + std::to_string(size) + ".csv", nx, ny);
		save_matrix_to_csv(v, "result/v-mpi-" + std::to_string(size) + ".csv", nx, ny);
		save_matrix_to_csv(p, "result/p-mpi-" + std::to_string(size) + ".csv", nx, ny);
	}

	MPI_Finalize();

	return 0;
}

