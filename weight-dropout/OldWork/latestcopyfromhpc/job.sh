#!/bin/bash

# execute in the general partition
#SBATCH --partition=general

# job name is my_job
#SBATCH --job-name=LMBP

# Kinda hacky. Basically, these subdepencies of libopenblas.so don't get included when we
# change the runpath, so we have to fix it for the executable
LD_PRELOAD=./lib64/libgfortran.so.5:./lib64/libquadmath.so.0

# load environment
source /opt/ohpc/admin/lmod/8.2.10/init/bash;

module load openmpi

make all

for workeri in {11..50}; do 
    echo "Running with worker: $workeri"
    #LD_PRELOAD=$LD_PRELOAD mpiexec -c $workeri lmbp-openmpi
    mpiexec -c $workeri lmbp-openmpi
done

