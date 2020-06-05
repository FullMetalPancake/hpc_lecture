#include <cassert>
#include <cstdio>
#include <chrono>
#include <vector>
#include "hdf5.h"
using namespace std;

int main (int argc, char** argv) {
  // grid of local spaces
  hsize_t dim[2] = {2, 2};
  
  // mpi communication
  int mpisize, mpirank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  
  // check if we have four different spaces
  assert(mpisize == dim[0]*dim[1]);
  
  // PURPLE HDF5 mpi initialization
  hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
  
  // BLUE HDF5 open file
  hid_t file = H5Fopen("data.h5", H5F_ACC_RDONLY, plist);
  
  // RED HDF5 link dataset of globalspace to file, so that result is correctly read
  hid_t dataset = H5Dopen(file, "dataset", H5P_DEFAULT);
  
  // GREEN HDF5 read 2d space from the write program
  // NxN for global space
  hid_t globalspace = H5Dget_space(dataset);
  
  // properties of the global spaces
  int ndim = H5Sget_simple_extent_ndims(globalspace);
  hsize_t N[ndim];
  H5Sget_simple_extent_dims(globalspace, N, NULL);
  hsize_t NX = N[0], NY = N[1];
  
  // properties of the four different local spaces EDITED FOR HOMEWORK
  hsize_t Nlocal[2] = {NX/dim[0], NY/dim[1]};
  hsize_t offset[2] = {mpirank / dim[0], mpirank % dim[0]};
  hsize_t count[2] = {Nlocal[0], Nlocal[1]};
  hsize_t stride[2] = {2,2};
  hsize_t block[2] = {1,1};
  
  
  // GREEN HDF5 read 2d space from the write program
  // NlocalxNlocal for local space
  hid_t localspace = H5Screate_simple(2, Nlocal, NULL);
  
  // GREEN HDF5 create tiles for reading in specified space
  H5Sselect_hyperslab(globalspace, H5S_SELECT_SET, offset, stride, count, block);
  
  // PURPLE HDF5 mpi continued
  H5Pclose(plist);
  plist = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
  
  // time computation
  auto tic = chrono::steady_clock::now();
  
  // RED HDF5 Read data
  vector<int> buffer(Nlocal[0]*Nlocal[1]);
  H5Dread(dataset, H5T_NATIVE_INT, localspace, globalspace, plist, &buffer[0]);
  
  auto toc = chrono::steady_clock::now();
  
  // RED HDF5 Close data
  H5Dclose(dataset);
  
  // GREEN HDF5 Close space
  H5Sclose(localspace);
  H5Sclose(globalspace);
  
  // BLUE HDF5 close file
  H5Fclose(file);
  
  // PURPLE HDF5 close mpi
  H5Pclose(plist);
  double time = chrono::duration<double>(toc - tic).count();
  
  // compute and print result
  int sum = 0;
//  for (int i=0; i<Nlocal[0]*Nlocal[1]; i++)
//    sum += buffer[i];
  for (int i=0; i<Nlocal[0]*Nlocal[1]; i++) {
      printf("%d ",buffer[i]);
      sum += buffer[i];
      if ((i + 1) % Nlocal[0] == 0) {
        printf("\n");
      }
  }
  
  printf("sum=%d\n",sum);
  printf("N=%d: %lf s (%lf GB/s)\n",NX*NY,time,4*NX*NY/time/1e9);
  MPI_Finalize();
}
