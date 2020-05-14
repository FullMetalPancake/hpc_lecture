#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=0:01:00
mpirun -bind-to core --mca mpi_cuda_support 0 -np $1 a.out
