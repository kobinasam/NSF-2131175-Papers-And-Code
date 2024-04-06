#!/bin/bash

make mp

for num_sample_time in {1..10}; do 
    for workeri in {4..50}; do 
        echo "Running with worker: $workeri, num_sample_time: $num_sample_time"
        ./lmbp-openmp $workeri $num_sample_time
    done
done

