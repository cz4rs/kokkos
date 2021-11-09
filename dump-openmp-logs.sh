#!/bin/bash 

# dump some openmp scheduling information
mkdir -p openmp-logs

# generate logs for a small number of iterations
OMP_PROC_BIND=spread OMP_PLACES=threads ./build/example/tutorial/scheduling_details/KokkosExample_scheduling_details -N 17 | tee openmp-logs/small-static.log
OMP_PROC_BIND=spread OMP_PLACES=threads ./build/example/tutorial/scheduling_details/KokkosExample_scheduling_details -N 17 -d | tee openmp-logs/small-dynamic.log

# generate logs for a big number of iterations
OMP_PROC_BIND=spread OMP_PLACES=threads ./build/example/tutorial/scheduling_details/KokkosExample_scheduling_details -N 1003 | tee openmp-logs/big-static.log
OMP_PROC_BIND=spread OMP_PLACES=threads ./build/example/tutorial/scheduling_details/KokkosExample_scheduling_details -N 1003 -d | tee openmp-logs/big-dynamic.log

