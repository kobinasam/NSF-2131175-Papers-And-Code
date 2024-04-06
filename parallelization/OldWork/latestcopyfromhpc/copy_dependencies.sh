#!/bin/bash

# execute in the general partition
#SBATCH --partition=general
 
# execute with 40 processes/tasks
#SBATCH --ntasks=1
 
# execute on 4 nodes
#SBATCH --nodes=1
 
# execute 4 threads per task
#SBATCH --cpus-per-task=4
 
# job name is my_job
#SBATCH --job-name=ParallelLM

 

THIRD_PARTY_DIR=./third-party
ARM_DIR=$THIRD_PARTY_DIR/armadillo-11.2.3/include
ARM_BITS_DIR=$THIRD_PARTY_DIR/armadillo-11.2.3/include/armadillo_bits
BLAS_DIR=$THIRD_PARTY_DIR/xianyi-OpenBLAS-b89fb70
LAPACK_DIR=$THIRD_PARTY_DIR/lapack-3.10.1
SYSTEM_LIBS=./lib64

INCLUDE_DIR=/tmp/include
LIB_DIR=/tmp/lib

#echo "Checking if system libs has anything before loading env";
#echo $SYSTEM_LIBS
#ls $SYSTEM_LIBS -p | grep fortran;
#ls /usr/lib64 -p | grep fortran;

# load environment
source /opt/ohpc/admin/lmod/8.2.10/init/bash

#echo "Checking if system libs has anything after loading env";
#echo $SYSTEM_LIBS
#ls $SYSTEM_LIBS -p | grep fortran;
#ls /usr/lib64 -p | grep fortran;

#for f in $BLAS_DIR/*; do
	#dir_length=${#BLAS_DIR};
	#echo copying $f to /tmp/lib/blas${f:$dir_length};
	##sbcast -f $f /tmp/include/armadillo_bits${f:$dir_length};
#done;

#srun mkdir -p $INCLUDE_DIR
#srun mkdir -p $LIB_DIR
#srun mkdir -p $INCLUDE_DIR/armadillo_bits/
#srun mkdir -p $LIB_DIR/blas/
#srun mkdir -p $LIB_DIR/lapack/
#srun mkdir -p $LIB_DIR/lib64/
#
#sbcast $ARM_DIR/armadillo $INCLUDE_DIR/armadillo
#for f in $(ls $ARM_BITS_DIR); do
#	echo copying $ARM_BITS_DIR/$f to /tmp/include/armadillo_bits/$f;
#	sbcast $ARM_BITS_DIR/$f /tmp/include/armadillo_bits/$f;
#done;
#
#for f in $(ls $LAPACK_DIR -p | grep -v /); do
#	echo copying $LAPACK_DIR/$f to /tmp/lib/lapack/$f;
#	sbcast -f $LAPACK_DIR/$f /tmp/lib/lapack/$f;
#done;
#
#for f in $(ls $BLAS_DIR -p | grep -v /); do
#	echo copying $BLAS_DIR/$f to /tmp/lib/blas/$f;
#	sbcast -f $BLAS_DIR/$f /tmp/lib/blas/$f;
#done;
#
for f in $(ls $SYSTEM_LIBS -p | grep -v /); do
	echo copying $SYSTEM_LIBS/$f to /tmp/lib/lib64/$f;
	sbcast -f $SYSTEM_LIBS/$f /tmp/lib/lib64/$f;
done;

