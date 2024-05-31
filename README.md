
# Accelerating RNN Controllers with Parallel Computing and Weight Dropout Techniques

This is a repository for my work using c++ to implement an RNN as a grid connected controller. The code is sitting in /main.

## Configuring Visual Studio:

I configured mine to use windows environment variables for my dependencies. Feel free to change this.

Edit the Project -> Project Properties -> Linker -> Input and replace Additional Dependencies with this:
```
	libopenblas.lib;msmpi.lib;$(CoreLibraryDependencies);%(AdditionalDependencies)
```

If machine is x64 machine:

- Edit the Project -> Project Properties -> C/C++ and replace Additional Include Directories with this:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$(MSMPI_INC);$(MSMPI_INC)\x64;$(ARMADILLO_INC);$(EIGEN_INC)```

- Edit the Project -> Project Properties -> Linker and replace Additional Library Directories with this:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$(MSMPI_LIB64);$(BLAS_LIB)```

- Edit the Project -> Project Properties -> Linker -> General and replace Additional Library Directories with this: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$(MSMPI_LIB64);$(BLAS_LIB)```

if machine is x86:

- Edit the Project -> Project Properties -> C/C++ and replace Additional Include Directories with this: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$(MSMPI_INC);$(MSMPI_INC)\x86;$(ARMADILLO_INC);$(EIGEN_INC)```

- Edit the Project -> Project Properties -> Linker and replace Additional Library Directories with this:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```$(MSMPI_LIB32);$(BLAS_LIB)```

- Edit the Project -> Project Properties -> Linker -> Input and replace Additional Dependencies: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```libopenblas.lib;msmpi.lib;$(CoreLibraryDependencies);%(AdditionalDependencies)```

## Dependencies:

### BLAS:

Download OpenBLAS from here: https://github.com/xianyi/OpenBLAS/releases

Extract the zip file to ```C:\OpenBlas```

Edit your environment variables to the variable "OpenBLAS" to point to the lib directory inside this zip file. e.g. ```OpenBLAS = C:\OpenBLAS\OpenBLAS-0.3.21-x64\lib```

### MPI:
Install Microsoft MPI from here: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi?redirectedfrom=MSDN

The website will ask you to download ```msmpisetup.exe``` and ```msmpisdk.msi```. Download both

Run ```msmpisetup.exe``` and install Microsoft MPI into ```C:\Program Files\Microsoft MPI```.

Run ```msmpisdk.msi``` and install the MPI SDK into ```C:\Program Files (x86)\Microsoft SDKs\MPI\```

The installers will add the environment variables for MSMPI above.

### Armadillo:

Instructions and documentation: https://arma.sourceforge.net/download.html

Download the latest armadillo version, unzip and put somewhere like ```C:\Armadillo```

Add an environment variable ```ARMADILLO_INC``` pointing to the Include directory, e.g. ```C:\Armadillo\armadillo-11.4.3\include```

### Eigen:

Download Eigen latest (or 3.4.0) from here: https://eigen.tuxfamily.org/index.php?title=Main_Page

Put into filesystem somewhere like ```C:/Eigen```

Create new environment variable EIGEN_INC and put the path to the whole *unzipped* directory, e.g. ```EIGEN_INC=C:\Eigen\eigen-3.4.0```

## Running the program:

You must have ```mpiexec``` on the PATH, which should have happened automatically if you correctly installed Microsoft MPI.

Also, your runtime linker must be able to find ```libopenblas.dll```. I was having trouble with this, so I copied this dll from ```$OpenBLAS``` directory to the same directory that lmbp.exe is placed, e.g. ```$(SourceDir)/x64/Debug/```.

You can run the script like this: ```mpiexec -n $1 lmbp.exe $2```, where ```$1``` is the parameter for number of workers for the parallelization and ```$2``` is the parameter for how many multiples of 10 trajectories to run for training.

The simplest execution to test if everything is working is thus: ```mpiexec -n 1 lmbp.exe 1.```

This repository mainly has three folders. 

We implemented A Novel Weight Dropout Approach to Accelerate the Neural Network Controller Embedded Implementation on FPGA for a Solar Inverter, Parallel Trajectory Training of Recurrent Neural Network Controllers with Levenberg–Marquardt and Forward Accumulation Through Time in Closed-loop Control Systems and lastly Accelerating RNN Controllers with Parallel Computing and Weight Dropout Techniques. 

Our main goal impementing the drop-out technique on our parallel trajectory training was to see how the implementation of the drop-out technique impacts the results of our trajectory parallelization. 

##Running the program on AWS:

First create a GitHub repository and commit the code to the repository

Now using Git Actions create a workflow to the AWS. 

To set up the workflow, click on “Actions” on your menu items under your repository name then click on “new workflow” and “set up a workflow yourself”. 

This is a sample “Action” which can be found on this link https://github.com/kobinasam/Trajectory-Parallelization-RNN/blob/main/.github/workflows/main.yml

You can download it and commit it to your repository as well. 

Now go to settings on the same repository. On the left side of the page scroll down to “Security” and then click on the submenu “Secrets and variables”.

Click on “Actions” and then “New repository secrets”

Define your name and secret {PUBLIC_PATH}, {HOSTIP }, {USER_NAME }, {PRIVATE_KEY} as strings as seen in the yml file. 

PUBLIC_PATH is the path to the root of your code
HOSTIP is the IP address given by AWS after your create an instance
USER_NAME is the name of the server ie, Ubuntu
PRIVATE_KEY is the private key given by AWS

After successfully setting up Git Actions, do a new commit to check if your code is directly deployed on your AWS server. 

To check if successful, open a new terminal then “ssh” to your AWS server using “ssh -i “permission-name” “server-name@private-key”

Change directory to check if the deployment was successfully done. 

##Dependancies:
Make sure to install all dependancies: MPI, Armadillo and Eigen. 

To run the code which is sitting in /main: 
First build the project using this script: mpic++ -I/usr/include/eigen3 -o myRNN RNN.cpp lmbp.cpp matrix_utility.cpp -larmadillo -llapack -lboost_iostreams -lmlpack

And then run this: mpiexec -n $1 lmbp.exe $2, where $1 is the parameter for number of workers for the parallelization and $2 is the parameter for how many multiples of 10 trajectories to run for training.

The simplest execution to test if everything is working is thus: mpiexec -n 1 lmbp.exe 1.

https://wiki.simcenter.utc.edu/doku.php/hpc_resources:how_to_run_jobs:login_node

