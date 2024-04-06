# include <iomanip>
# include <iostream>
# include <mpi.h>

# include <armadillo>

using namespace std;
using namespace arma;

int main ( int argc, char *argv[] );

// ==================================================================
int main ( int argc, char *argv[] )
{
    int id;
    int ierr;
    int total_workers;
    double wtime;
 
    ierr = MPI_Init ( NULL, NULL );
    if ( ierr != 0 )
    {
        cout << "\n";
        cout << "HELLO_MPI - Fatal error!\n";
        cout << "  MPI_Init returned nonzero ierr.\n";
        exit ( 1 );
    }
    
    ierr = MPI_Comm_size ( MPI_COMM_WORLD, &total_workers );
    ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &id );

    if (id == 0)
    {
        cout << "running with " << total_workers << endl;
    }

    cout << "Worker " << id << " ready!" << endl;

    mat A(5, 5, fill::randu);
    mat X = A.t()*A;
    
    mat R1 = chol(X);
    mat R2 = chol(X, "lower");

    MPI_Finalize();
    return 0;
}

 
