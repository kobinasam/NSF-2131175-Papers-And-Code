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

if [[ -f basic-openmpi ]]
then
    rm basic-openmpi
fi

# It compiles just fine but wont execute. 
#It fucking won't work with openblas. If I don't add the LD_PRELOAD, it complains about not having the dependency "libgfortran.so.5"
# of the dependency libopenblas.so sitting in third-party/xianyi...
# if I do include LD_PRELOAD, then it gives this generic unhelpful error about PMIX or some shit
#mpic++ -I ./third-party/armadillo-11.2.3/include -Wl,-rpath-link=./lib64 -Wl,-rpath=./lib64 -L ./third-party/xianyi-OpenBLAS-b89fb70/ -lopenblas -o basic-openmpi basic-openmpi.cpp
#LD_PRELOAD=./lib64/libgfortran.so.5:./lib64/libquadmath.so.0
#LD_PRELOAD=$LD_PRELOAD mpiexec -c 2 basic-openmpi

mpic++ -I ./third-party/armadillo-11.2.3/include -L ./third-party/BLAS-3.10.0/ -L ./third-party/lapack-3.10.1/ -llapack -lblas -lgfortran -o basic-openmpi basic-openmpi.cpp
mpiexec -c 2 basic-openmpi

#/home/jsturtz/LCL_690_1200_parallel_10_100_cpp/third-party[jsturtz@firefly BLAS-3.10.0]$ ls ./blas_LINUX.a
#./blas_LINUX.a
#mpic++ -std=c++17 -O3 -Wl,-rpath-link=./lib64 -Wl,-rpath=./lib64 -I ./third-party/armadillo-11.2.3/include -L ./third-party/xianyi-OpenBLAS-b89fb70/ -L ./lib64 -lopenblas -o lmbp-openmpi lmbp-openmpi.cpp
