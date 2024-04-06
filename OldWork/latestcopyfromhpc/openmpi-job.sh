#!/bin/bash

# For some reason headnode won't run the build process? wtf?
#make mpi

module load openmpi

for num_sample_time in {1..10}; do 
    for workeri in {1..50}; do 
        echo "Running with worker: $workeri, num_sample_time: $num_sample_time"
        mpirun -c $workeri lmbp-openmpi $num_sample_time
    done
done

