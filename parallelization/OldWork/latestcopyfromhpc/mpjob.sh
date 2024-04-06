
make mp

for workeri in {1..50}; do 
    for num_sample_time in {1..10}; do
        echo "Running with worker: $workeri, sampletime: $num_sample_time"
        lmbp-openmpi $workeri $num_sample_time
    done
done

