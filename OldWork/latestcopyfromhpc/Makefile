
MPICC=mpic++
CC=g++

LIB64=./lib64
THIRD_PARTY=./third-party
ARMADILLO=$(THIRD_PARTY)/armadillo-11.2.3/
LAPACKE=$(THIRD_PARTY)/lapack-3.10.1/
BLAS=$(THIRD_PARTY)/xianyi-OpenBLAS-b89fb70/
EIGEN=$(THIRD_PARTY)/eigen-3.3.9/

CFLAGS=-std=c++17 \
       -O3 \
       -Wl,-rpath-link=./lib64 \
	   -Wl,-rpath=./lib64 \
       -I ${ARMADILLO}include \
       -I ${EIGEN} \
       -L $(LAPACKE) \
       -L $(BLAS) \
       -L $(LIB64)

MPCFLAGS=-fopenmp

LD_FLAGS=-lopenblas

OPENMPI=lmbp-openmpi
OPENMP=lmbp-openmp
TESTS=tests
SANDBOX=sandbox

all: mpi mp tests sandbox

sandbox: $(SANDBOX).cpp
	  $(CC) $(CFLAGS) $(LD_FLAGS) -o $(SANDBOX) $(SANDBOX).cpp

tests: $(TESTS).cpp
	  $(CC) $(CFLAGS) $(LD_FLAGS) -o $(TESTS) $(TESTS).cpp

mp: $(OPENMP).cpp
	  $(CC) $(MPCFLAGS) $(CFLAGS) $(LD_FLAGS) -o $(OPENMP) $(OPENMP).cpp

mpi: $(OPENMPI).cpp
	  $(MPICC) $(CFLAGS) $(LD_FLAGS) -o $(OPENMPI) $(OPENMPI).cpp

clean:
ifneq ("$(wildcard $(OPENMP))","")
	rm $(OPENMP)
endif
ifneq ("$(wildcard $(OPENMPI))","")
	rm $(OPENMPI)
endif
