# include <cstdlib>
# include <ctime>
# include <iomanip>
# include <iostream>
# include <mpi.h>
# include <map>
# include <vector>

# include <armadillo>

using namespace std;
using namespace arma;

typedef std::vector<double> stdvec;
typedef std::vector< std::vector<double> > stdvecvec;

int main ( int argc, char *argv[] );

// ==================================================================
// Utility functions start here

// ==================================================================
void print_mat(const mat m, const string &name, bool hide_contents=false, double precision=16)
{
    std::cout << "============================================" << endl;
    std::cout << "Name: " << name << ", Rows: " << m.n_rows << ", Cols: " << m.n_cols << endl;
    if (!hide_contents)
    {
        for (uword i = 0; i < m.n_rows; i++)
        {
            for (uword j = 0; j < m.n_cols; j++)
            {
                std::cout << setprecision(precision) << m.at(i, j) << " ";
            }
            std::cout << endl;
        }
    }
    if (m.is_empty())
    {
        std::cout << name << " is empty!";
    }
    std::cout << endl;
}

// ==================================================================
pair<int, int> end_start_elems_to_process(int id, int num_elements, int num_groups)
{
    int last_idx = (id + 1) * num_elements / num_groups;
    int first_idx = id * num_elements / num_groups;
    return {first_idx, last_idx};
}

// ==================================================================
int num_elems_to_process(int id, int num_elements, int num_groups)
{
    if (id >= num_elements)
    {
        return 0;
    }

    pair<int, int> end_start = end_start_elems_to_process(id, num_elements, num_groups);
    return end_start.second - end_start.first;    
}

// ==================================================================
stdvecvec mat_to_std_vec(const mat &M) 
{
    stdvecvec V(M.n_rows);
    for (size_t i = 0; i < M.n_rows; ++i) {
        V[i] = arma::conv_to<stdvec>::from(M.row(i));
    };
    return V;
}

// ==================================================================
void mat_to_std_array(const mat &M, double *arr)
{
    // NOTE: Requires arr to be allocated correctly same rows / cols as M

    // FIXME: Is there a better way to convert from armadillo objects to arrays?
    // Looks like armadillo only provides methods to convert to vectors.
    // Assuming that conversion is efficient, probably best to convert arma::Mat -> vector -> array?
    stdvecvec V = mat_to_std_vec(M);

    int numrows = V.size();
    int numcols = V[0].size();    
    for (int i = 0; i < numrows; i++)
    {
        for (int j = 0; j < numcols; j++)
        {
            arr[i * numcols + j] = V[i][j];    
        }
    }
}

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

    const int trajectory_length=1000;
    const int num_weights=86;
    int num_samples=100;

    // We have no need for more workers than we have samples, so use active_workers
    int active_workers = min(num_samples, total_workers);

    // Used to distingish messages from workers to master
    int J_TOTAL_TAG=0;
    int J_MATRIX_TAG=1;
    int E_HIST_TAG=2;

    // each worker computes how many elements it needs to process    
    int samples_to_process = num_elems_to_process(id, num_samples, active_workers);
    cout << "samples to process: " << samples_to_process << endl;
    
    // some workers may have nothing to do (if workerid >= num_samples)
    if (samples_to_process > 0)
    {
        // Each worker does whatever work it needs to do to compute the three values for each element
        double j_total = 1;
        rowvec e_hist_err = rowvec(samples_to_process * trajectory_length, arma::fill::ones) * id;
        mat j_matrix = mat(samples_to_process * trajectory_length, num_weights, arma::fill::ones) * id;
        
        if (id == 0)
        {
            double j_total_sum = 0.0;

            rowvec hist_err_total = rowvec(num_samples * trajectory_length, arma::fill::zeros);
            mat j_matrix_total = mat(num_samples * trajectory_length, num_weights, arma::fill::zeros);

            // The master first adds its piece to the aggregation variables, then waits to get chunks from the workers
            j_total_sum += j_total;
            hist_err_total.cols(0, samples_to_process * trajectory_length - 1) = e_hist_err;
            j_matrix_total.rows(0, samples_to_process * trajectory_length - 1) = j_matrix;

            // The master needs to know the starting indices for where to put worker data
            int rolling_sum = 0;
            map<int, int> starting_indices;
            for (int i = 0; i < active_workers-1; i++)
            {
                rolling_sum += num_elems_to_process(i, num_samples, active_workers);
                starting_indices[i+1] = rolling_sum;
            }

            MPI_Status status;

            // Use one buffer each for these messages from each worker
            double *sender_hist_err = new double[trajectory_length * num_samples];
            double *sender_j_matrix = new double[trajectory_length * num_samples * num_weights];

            for (int i=0; i < active_workers - 1; i++)
            {
                // The first thing any worker sends is the j_total, which is a single value
                MPI_Recv(&j_total, 1, MPI_DOUBLE, MPI_ANY_SOURCE, J_TOTAL_TAG, MPI_COMM_WORLD, &status);
                j_total_sum += j_total;

                // That same worker that sent the j_total will immediately send the arrays
                // We can know how many elements they intend to send since we know their id
                int senderid = status.MPI_SOURCE;
                int sender_samples = num_elems_to_process(senderid, num_samples, active_workers);

                MPI_Recv(sender_hist_err, trajectory_length * sender_samples, MPI_DOUBLE, senderid, E_HIST_TAG, MPI_COMM_WORLD, NULL);
                MPI_Recv(sender_j_matrix, trajectory_length * sender_samples * num_weights, MPI_DOUBLE, senderid, J_MATRIX_TAG, MPI_COMM_WORLD, NULL);

                int start = starting_indices[senderid] * trajectory_length;
                int end = start + (sender_samples * trajectory_length) - 1;
                
                hist_err_total.cols(start, end) = rowvec(&sender_hist_err[0], trajectory_length * sender_samples, false);
                j_matrix_total.rows(start, end) = mat(&sender_j_matrix[0], trajectory_length * sender_samples, num_weights, false);

            }

            // Then make sure to free the allocation
            delete[] sender_hist_err;
            delete[] sender_j_matrix;

            cout << "J_total_Sum: " << j_total_sum << endl;
            cout << "hist_err_shape: (" << hist_err_total.n_rows << ", " << hist_err_total.n_cols << ")" << endl;
            cout << "j_matrix shape: (" << j_matrix_total.n_rows << ", " << j_matrix_total.n_cols << ")" << endl;

            print_mat(hist_err_total.cols(0, 4), "hist err total");
            print_mat(hist_err_total.cols(1000, 1004), "hist err total");
            print_mat(hist_err_total.cols(2000, 2004), "hist err total");
            print_mat(hist_err_total.cols(3000, 3004), "hist err total");
            print_mat(hist_err_total.cols(4000, 4004), "hist err total");
            print_mat(hist_err_total.cols(5000, 5004), "hist err total");
            print_mat(hist_err_total.cols(6000, 6004), "hist err total");
            print_mat(hist_err_total.cols(7000, 7004), "hist err total");
            print_mat(hist_err_total.cols(8000, 8004), "hist err total");
            print_mat(hist_err_total.cols(9000, 9004), "hist err total");

            print_mat(j_matrix_total.rows(0, 4).cols(0, 4), "hist err total");
            print_mat(j_matrix_total.rows(1000, 1004).cols(0, 4), "j matrix");
            print_mat(j_matrix_total.rows(2000, 2004).cols(0, 4), "j matrix");
            print_mat(j_matrix_total.rows(3000, 3004).cols(0, 4), "j matrix");
            print_mat(j_matrix_total.rows(4000, 4004).cols(0, 4), "j matrix");
            print_mat(j_matrix_total.rows(5000, 5004).cols(0, 4), "j matrix");
            print_mat(j_matrix_total.rows(6000, 6004).cols(0, 4), "j matrix");
            print_mat(j_matrix_total.rows(7000, 7004).cols(0, 4), "j matrix");
            print_mat(j_matrix_total.rows(8000, 8004).cols(0, 4), "j matrix");
            print_mat(j_matrix_total.rows(9000, 9004).cols(0, 4), "j matrix");

        }

        // The workers have done their work, so they just send to the master
        else
        {
            double *sender_hist_err = new double[samples_to_process * trajectory_length];
            double *sender_j_matrix = new double[samples_to_process * trajectory_length * num_weights];

            mat_to_std_array(e_hist_err, sender_hist_err);
            mat_to_std_array(j_matrix, sender_j_matrix);

            MPI_Send(&j_total, 1, MPI_DOUBLE, 0, J_TOTAL_TAG, MPI_COMM_WORLD);
            MPI_Send(sender_hist_err, trajectory_length * samples_to_process, MPI_DOUBLE, 0, E_HIST_TAG, MPI_COMM_WORLD);
            MPI_Send(sender_j_matrix, trajectory_length * samples_to_process * num_weights, MPI_DOUBLE, 0, J_MATRIX_TAG, MPI_COMM_WORLD);

            delete[] sender_hist_err;
            delete[] sender_j_matrix;
        }
    }

    MPI_Finalize();
    return 0;
}

 
