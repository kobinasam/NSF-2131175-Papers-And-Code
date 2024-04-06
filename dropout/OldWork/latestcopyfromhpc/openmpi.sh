#!/bin/bash
 
# execute in the general partition
#SBATCH --partition=general

# execute with 40 processes/tasks
#SBATCH --ntasks=2
 
# execute on 4 nodes
#SBATCH --nodes=1
 
## execute 4 threads per task
##SBATCH --cpus-per-task=4
 
# job name is my_job
#SBATCH --job-name=TEST
 
# load environment
source /opt/ohpc/admin/lmod/8.2.10/init/bash
module load openmpi

mpic++ -std=c++17 -O3 -Wl,-rpath-link=./lib64 -Wl,-rpath=./lib64 -I ./third-party/armadillo-11.2.3/include -L ./third-party/lapack-3.10.1/ -L ./third-party/xianyi-OpenBLAS-b89fb70/ -L ./lib64 -lopenblas -o test_openmpi test_openmpi.cpp

LD_PRELOAD=./lib64/libgfortran.so.5:./lib64/libquadmath.so.0
LD_PRELOAD=$LD_PRELOAD mpiexec -c 2 test_openmpi

