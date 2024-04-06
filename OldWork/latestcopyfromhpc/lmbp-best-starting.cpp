
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <math.h>
#include <map>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <utility>
#include <unordered_set>
#include <map>
#include <vector>

#include <mpi.h>
#include <armadillo>

using namespace std;
using namespace arma;

typedef std::vector<double> stdvec;
typedef std::vector< std::vector<double> > stdvecvec;

#define _USE_MATH_DEFINES

#undef ARMA_OPENMP_THREADS
#define ARMA_OPENMP_THREADS 50 // Max threads used by armadillo for whatever logic they have parallelized. Cool
#define ARMA_NO_DEBUG          // Only define this if we want to disable debugging for production code

// Used to distingish message types in sending/receiving messages
#define MPI_J_TOTAL 0
#define MPI_NEW_JOB 1
#define MPI_JOB_FINISHED 2
#define MPI_LOOP_DONE 3
#define MPI_SOLUTION_FOUND 4

mat preloaded_idq_refs;

// ==================================================================
// Utility functions start here
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
double same_within_precision(const mat &m1, const mat &m2)
{
    double precision = 1e-16;
    while (!approx_equal(m1, m2, "both", precision, precision) && precision < 1)
    {
        precision = precision * 10;
    }
    return precision;
}

// ==================================================================
// Floating point math sucks. Use this to guarantee you can round up
// or down on a floating point value
float round_to_decimal(float var, int dec)
{
    int multiplier = pow(10, dec);
    int value = (int)(var * multiplier + .5);
    return (float)value / multiplier;
}

// ==================================================================
void save_matrix_to_csv(const mat &M, string filename)
{
    ofstream file_handle;
    file_handle.open(filename);
    if (!file_handle.good())
    {
        throw std::runtime_error("Problem writing to file: " + filename);
    }
    M.save(file_handle, arma::csv_ascii);
    file_handle.close();
}

// ==================================================================
void load_matrix_from_csv(mat &M, string filename)
{
    /* cout << "Loading this file: " << filename << endl; */
    ifstream file_handle;
    file_handle.open(filename);
    if(file_handle.fail())
    {
        throw std::runtime_error("Failed to open " + filename + " due to reason: " + strerror(errno));
    }
    M.load(file_handle, arma::csv_ascii);
    file_handle.close();
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
    if (M.n_elem == 0 || M.n_rows == 0 || M.n_cols == 0)
    {
        throw ("Called mat_to_std_vec with incorrect shape");
    }

    stdvecvec V(M.n_rows);
    for (size_t i = 0; i < M.n_rows; ++i) {
        V[i] = arma::conv_to<stdvec>::from(M.row(i));
    };
    return V;
}

// ==================================================================
void mat_to_std_array(const mat* M, double *arr)
{
    // NOTE: Requires arr to be allocated correctly same rows / cols as M

    // FIXME: Is there a better way to convert from armadillo objects to arrays?
    // Looks like armadillo only provides methods to convert to vectors.
    // Assuming that conversion is efficient, probably best to convert arma::Mat -> vector -> array?
    stdvecvec V = mat_to_std_vec(*M);

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
void mats_to_std_array(vector<mat*> Ms, double *arr)
{
    // Combines all matrices in Ms into a single flattened array
    for (int i=0; i < Ms.size(); i++)
    {
        mat_to_std_array(Ms[i], arr);
        // Use pointer arithmetic to increment pointer to fill the right spot in the buffer
        arr += Ms[i]->n_elem;
    }
}

// ==================================================================
void std_array_to_mat(double* arr, mat &M)
{
    M = mat(arr, M.n_cols, M.n_rows).t();
}

// ==================================================================
void std_array_to_mats(double* arr, vector<mat*> Ms)
{
    // Fill matrices with values from array
    int arr_idx = 0; 
    for (int i=0; i < Ms.size(); i++)
    {
        std_array_to_mat(arr, *(Ms[i]));
        arr = arr + Ms[i]->n_elem;
    }
}

// ==================================================================
int count_in_matrices(vector<mat*> matrices)
{
    int total_count=0;
    for (int i=0; i < matrices.size(); i++)
    {
        int count = matrices[i]->n_elem;
        total_count += count;
    }
    return total_count;
}

// ==================================================================
void send_matrices(vector<mat*> matrices, vector<int> &receivers, int tag)
{
    // We don't want to make multiple calls to send multiple matrices and we don't want to convert multiple matrices
    // multiple times to arrays. So we have this function to send multiple matrices to multiple senders to consolidate work
    // The caller just packages the pointers to the matrices into a vector
    // Then, this function iterates over those matrices to figure out how the total count of the data to send
    // Then we can know how to call one MPI_Send() rather than N sends for N matrices

    int total_count=count_in_matrices(matrices);
    double *payload = new double[total_count];
    mats_to_std_array(matrices, payload);
    for (int i = 0; i < receivers.size(); i++)
    {
        MPI_Send(payload, total_count, MPI_DOUBLE, receivers[i], tag, MPI_COMM_WORLD);
    }

    delete[] payload;
}

// ==================================================================
void send_matrices(vector<mat*> matrices, int receiver, int tag)
{
    vector<int> receivers = {receiver};
    send_matrices(matrices, receivers, tag);
}

// ==================================================================
MPI_Status recv_matrices(vector<mat*> matrices, int sender=MPI_ANY_SENDER, int tag=MPI_ANY_TAG)
{
    // Like with send_matrices, the idea is to provide a vector of matrices
    // (the size of each matrix determined by the caller) and then we just fill those matrices
    // with the values from the sender
    int total_count=count_in_matrices(matrices);
    double *payload = new double[total_count];

    MPI_Status status;
    MPI_Recv(payload, total_count, MPI_DOUBLE, sender, tag, MPI_COMM_WORLD, &status);

    std_array_to_mats(payload, matrices);
    delete[] payload;
    return status;
}

// ======================================================================
// Business logic functions here

// ======================================================================
// Implements calculateIdq_ref.m
colvec calculate_idq_ref(
    const int &trajectory_number,
    const int &time_step,
    const double &ts)
{
    // NOTE: trajectory_number and time_step come from zero-indexed, rather than one-indexed code
    // So when comparing to matlab, we must add an extra value
    float period = 0.1;
        int change_number = floor(round_to_decimal(time_step * ts / period, 2)) + 1;

    // Well the seed computation is inconsistent.
    colvec idq_ref;
    int seed = ((trajectory_number+1) % 10) * 10000 + change_number;
    srand(seed);
    arma_rng::set_seed(seed);
    idq_ref = colvec({ -300, -300 }) + colvec({ 600, 360 }) % mat(2, 1, arma::fill::randu);
    idq_ref = arma::round(idq_ref * 10) / 10;

    // The calculation for the seed is all fucked, so just use the values from matlab
    idq_ref = preloaded_idq_refs.rows(time_step*2, time_step*2+1).col(trajectory_number);
    return idq_ref;
}

// ======================================================================
mat exdiag(const mat &x)
{
    mat y = mat(x.n_rows, x.n_elem, arma::fill::zeros);
    for (uword i = 0; i < x.n_rows; i++)
    {
        y.row(i).cols(x.n_cols * i, x.n_cols * (i+1) - 1) = x.row(i);
    }
    return y;
}

// ======================================================================
void net_action(
    const colvec &idq,
    const colvec &idq_ref,
    const mat &hist_err,
    const mat &w1,
    const mat &w2,
    const mat &w3,
    const bool flag,
    const bool use_shortcuts,
    const bool use_idq,
    const double gain1,
    const double gain2,
    const double gain3,
    mat &o3,
    mat &dnet_dw,
    mat &dnet_didq,
    mat &dnet_dhist_err
    )

{
    colvec input0A = (idq / gain1);
    colvec output0A = input0A.transform([](double val) { return tanh(val); } );
    colvec input0B = (idq.rows(0, 1) - idq_ref) / gain2;
    colvec output0B = input0B.transform([](double val) { return tanh(val); } );
    colvec input0C = hist_err / gain3;
    colvec output0C = input0C.transform([](double val) { return tanh(val); } );

    colvec input1 = use_idq ? join_vert(output0A, output0B, output0C, colvec({-1})): join_vert(output0B, output0C, colvec({-1}));

    colvec sum1 = w1 * input1;
    colvec o1 = sum1.transform([](double val) { return tanh(val); } );

    colvec input2 = use_shortcuts ? join_vert(o1, input1): join_vert(o1, colvec({-1}));
    colvec sum2 = w2 * input2;
    colvec o2 = sum2.transform([](double val) { return tanh(val); } );

    colvec input3 = use_shortcuts ? join_vert(o2, input2): join_vert(o2, colvec({-1}));
    colvec sum3 = w3 * input3;
    o3 = sum3.transform([](double val) { return tanh(val); } );

    mat do3_dw3, do3_do2, do3_dw2, do2_dw2, do2_do1, do1_dw1, do3_do1, do3_dw1, do3_do1_d3;
    mat dinput1_0A_0B_didq, do1_dinput1_0A_0B, do3_dinput1_0A_0B_d3, do3_dinput1_0A_0B;
    mat dinput1_0C_dhist_err, do1_dinput1_0C, do3_dinput1_0C, do2_dinput1_0A_0B_d2, do3_dinput1_0A_0B_d2;
    mat do3_dinput1_0C_d3, do2_dinput1_0C_d2, do3_dinput1_0C_d2;

    if (flag)
    {
        // compute Dnet_Dw
        // third layer
        do3_dw3 = (1 - o3 % o3) * input3.t();
        dnet_dw = exdiag(do3_dw3);

        // second layer
        do3_do2 = diagmat(1 - o3 % o3) * w3.cols(0, w2.n_rows - 1);
        do2_dw2 = exdiag((1 - o2 % o2) * input2.t());

        do3_dw2 = do3_do2 * do2_dw2;
        dnet_dw = join_horiz(dnet_dw, do3_dw2);

        // first layer
        do2_do1 = diagmat(1 - o2 % o2) * w2.cols(0, w1.n_rows - 1);

        if (use_shortcuts)
        {
            do3_do1_d3 = diagmat(1 - o3 % o3) * w3.cols(w2.n_rows, w2.n_rows + w1.n_rows - 1);
            do3_do1 = do3_do1_d3 + do3_do2 * do2_do1;
        }
        else
        {
            do3_do1 = do3_do2 * do2_do1;
        }

        do1_dw1 = exdiag((1 - o1 % o1) * input1.t());
        do3_dw1 = do3_do1 * do1_dw1;

        dnet_dw = join_horiz(dnet_dw, do3_dw1);

        if (use_idq)
        {
            dinput1_0A_0B_didq = join_vert(
                diagmat((1-output0A % output0A) / gain1),
                diagmat((1-output0B % output0B) / gain2),
                mat(2, 4, arma::fill::zeros)
            );
        }
        else
        {
            dinput1_0A_0B_didq = diagmat((1-output0B % output0B) / gain2);
        }
        do1_dinput1_0A_0B = diagmat(1 - o1 % o1) * w1.cols(0, w1.n_cols - 4);

        // compute Dnet_Didq
        if (use_shortcuts)
        {
            do3_dinput1_0A_0B_d3 = diagmat(1 - o3 % o3) * w3.cols(w2.n_rows + w1.n_rows, w3.n_cols - 4);
            do2_dinput1_0A_0B_d2 = diagmat(1 - o2 % o2) * w2.cols(w1.n_rows, w2.n_cols - 4);
            do3_dinput1_0A_0B_d2 = do3_do2 * do2_dinput1_0A_0B_d2;
            do3_dinput1_0A_0B = do3_do1 * do1_dinput1_0A_0B + do3_dinput1_0A_0B_d3 + do3_dinput1_0A_0B_d2;
        }
        else
        {
            do3_dinput1_0A_0B = do3_do1 * do1_dinput1_0A_0B;
        }

        dnet_didq = do3_dinput1_0A_0B * dinput1_0A_0B_didq;

        // compute dnet_dhist_err
        dinput1_0C_dhist_err = diagmat((1 - output0C % output0C) / gain3);
        do1_dinput1_0C = diagmat(1 - o1 % o1) * w1.cols(w1.n_cols - 3, w1.n_cols - 2);

        if (use_shortcuts)
        {
            do3_dinput1_0C_d3 = diagmat(1 - o3 % o3) * w3.cols(w3.n_cols - 3, w3.n_cols - 2);
            do2_dinput1_0C_d2 = diagmat(1 - o2 % o2) * w2.cols(w2.n_cols - 3, w2.n_cols - 2);
            do3_dinput1_0C_d2 = do3_do2 * do2_dinput1_0C_d2;
            do3_dinput1_0C = do3_do1 * do1_dinput1_0C + do3_dinput1_0C_d3 + do3_dinput1_0C_d2;
        }
        else
        {
            do3_dinput1_0C = do3_do1 * do1_dinput1_0C;
        }
        dnet_dhist_err = do3_dinput1_0C * dinput1_0C_dhist_err;
    }
}

// ============================================================================
// Implementation of unrollTrajectoryFull.m
void unroll_trajectory_full(
    const colvec &initial_idq,
        const colvec &vdq,
    const int &trajectory_number,
    const int &trajectory_length,
    const mat &w3,
    const mat &w2,
    const mat &w1,
    const bool use_idq,
    const bool use_shortcuts,
    const mat &a,
    const mat &b,
    const double &vmax,
    const double &ts,
    const double &gain1,
    const double &gain2,
    const double &gain3,
    const double &cost_term_power,
    double &j_total,
    rowvec &e_hist_err,
    mat &j_matrix,
    mat &idq_his,
    mat &idq_ref_his)
{
    colvec idq = initial_idq;
    double num_weights = w1.size() + w2.size() + w3.size();

    idq_his = mat(6,trajectory_length, arma::fill::zeros);
    idq_ref_his = mat(2, trajectory_length, arma::fill::zeros);
    mat hist_err = mat(2, trajectory_length, arma::fill::zeros);
    e_hist_err = rowvec(trajectory_length, arma::fill::zeros);

    mat didq_dw, dvdq_dw, didq_dw_matrix_sum;
    didq_dw = mat(6, num_weights, arma::fill::zeros);
    dvdq_dw = mat(2, num_weights, arma::fill::zeros);
    j_matrix = mat(trajectory_length+1, num_weights, arma::fill::zeros);
    didq_dw_matrix_sum = mat(2, num_weights, arma::fill::zeros);

    mat err_integral, dudq_dw;
    colvec idq_ref, idq_refi;

    // outputs of net_action
    mat o3, udq, ndq, dnet_dw, dnet_didq, dnet_dhist_err;

    for (int i = 0; i < trajectory_length; i++)
    {
        err_integral = ts * (arma::sum(hist_err, 1) - hist_err.col(i) / 2.0);
        idq_ref = calculate_idq_ref(trajectory_number, i, ts);
        idq_ref_his.col(i) = idq_ref;

                hist_err.col(i) = (idq.rows(0, 1) - idq_ref); // when the error is too small, the calculation becomes inaccurate
                e_hist_err.col(i) = pow((arma::sum(hist_err.col(i) % hist_err.col(i))), cost_term_power / 2.0);// the calculation process of pow is slightly from that in Matlab
                net_action(idq, idq_ref, err_integral, w1, w2, w3, false, use_shortcuts, use_idq, gain1, gain2, gain3, ndq, dnet_dw, dnet_didq, dnet_dhist_err);
                udq = ndq * vmax;
                net_action(idq, idq_ref, err_integral, w1, w2, w3, true, use_shortcuts, use_idq, gain1, gain2, gain3, o3, dnet_dw, dnet_didq, dnet_dhist_err);
        if (use_idq)
        {
            dudq_dw = vmax * (dnet_dw + dnet_didq * didq_dw + dnet_dhist_err * ts * didq_dw_matrix_sum);
        }
        else
        {
            dudq_dw = vmax * (dnet_dw + dnet_didq * didq_dw.rows(0, 1) + dnet_dhist_err * ts * (didq_dw_matrix_sum - didq_dw.rows(0, 1) / 2.0));
        }

        didq_dw = a * didq_dw + b * join_vert(dvdq_dw, dudq_dw, dvdq_dw);
        didq_dw_matrix_sum = didq_dw_matrix_sum + didq_dw.rows(0, 1);
        idq = a * idq + b * join_vert(vdq, udq, colvec({0, 0}));

        // add saturation to dq currents
        // idq(1:5) = max(min(idq(1:4), 1000*ones(4,1)),-1000*ones(4,1));
        idq.rows(0, 3) = arma::max(arma::min(idq.rows(0, 3), 1000*mat(4, 1, arma::fill::ones)), -1000 * mat(4, 1, arma::fill::ones));

        idq_his.col(i) = idq;
        idq_refi = calculate_idq_ref(trajectory_number, i+1, ts);
        j_matrix.row(i+1) = (idq.rows(0, 1) -idq_refi).t() * didq_dw.rows(0, 1) * cost_term_power * pow(arma::sum((idq.rows(0, 1) - idq_refi) % (idq.rows(0, 1) - idq_refi)), cost_term_power / 2.0 - 1);
    }

    j_total = arma::sum(e_hist_err % e_hist_err);
    j_matrix.shed_row(j_matrix.n_rows-1);
}

// ============================================================================
int main ( int argc, char *argv[] )
{
    int rank;
    int mpi_err;
    int total_workers;

    mpi_err = MPI_Init ( NULL, NULL);
    if ( mpi_err != 0 )
    {
        cout << endl;
        cout << "HELLO_MPI - Fatal error!" << endl;
        cout << "MPI_Init returned nonzero error." << endl;
        exit ( 1 );
    }

    mpi_err = MPI_Comm_size ( MPI_COMM_WORLD, &total_workers );
    mpi_err = MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

    const bool use_shortcuts = false;
    const bool use_idq = false;
    const double vd = 690;
    const colvec vdq = colvec({vd, 0.0});
    
    const double t_final = 1;
    const double ts = 0.001;
    const double vdc = 1200;
    const colvec idq_ref_centre = colvec({0, 0});
    
    const double vmax = vdc * sqrt(3.0 / 2.0) / 2.0;
    
    const double gain1 = 1000.0;
    const double gain2 = 100*1;
    const double gain3 = 100*1;
    
    const double cost_term_power = 1.0 / 2.0;
    
    double mu;
    const double mu_dec = 0.1;
    const double mu_inc = 10;
    const double mu_max = 1e10;
    const double mu_min = 1e-20;
    
    const int trajectory_length = t_final / ts;
    
    const mat idq_start_positions = {
        { 0.475860587278343, -0.002854960144321, -0.777698538311603, 0.879677164233489, -1.314723503486884, 0.064516742311104, -0.037533188819172, -1.725427789528692, 0.093108760058804, -0.430206242426100 },
        { 1.412232686444504, 0.919867079806395, 0.566696097539305, 2.038876251414042, -0.416411219699434, 0.600291949185784, -1.896304493622450, 0.288228089665011, -0.378157056589758, -1.627322736469780 },
        { 0.022608484309598, 0.149808732632761, -1.382621159480352, 0.923932448688937, 1.224687824785337, -1.361514954864567, -2.127976768182674, -1.594183720266807, -1.482676111059003, 0.166347492460066 },
        { -0.047869410220206, 1.404933445676977, 0.244474675589888, 0.266917446595828, -0.043584205546333, 0.347592631960065, -1.176923330714958, 0.110218849223381, -0.043818585358295, 0.376265910450719 },
        { 1.701334654274959, 1.034121539569710, 0.808438803167691, 0.641661506421779, 0.582423277447969, -0.181843218459334, -0.990532220484176, 0.787066676357980, 0.960825211682115, -0.226950464706233 },
        { -0.509711712767427, 0.291570288770806, 0.213041698417000, 0.425485355625296, -1.006500074619336, -0.939534765941492, -1.173032327267405, -0.002226786313836, 1.738244932613340, -1.148912289618790 }
    };
    
    const mat a = {
        { 0.922902679404235, 0.365403020170600, 0.001311850123628, 0.000519398207289, -0.006031602076712, -0.002388080200093},
        { -0.365403020170600, 0.922902679404235, -0.000519398207289, 0.001311850123628, 0.002388080200093, -0.006031602076712},
        { 0.001311850123628, 0.000519398207289, 0.922902679404235, 0.365403020170600, 0.006031602076712, 0.002388080200093},
        { -0.000519398207289, 0.001311850123628, -0.365403020170600, 0.922902679404235, -0.002388080200093, 0.006031602076712},
        { 0.120632041534246, 0.047761604001858, -0.120632041534246, -0.047761604001858, 0.921566702872299, 0.364874069642510},
        { -0.047761604001858, 0.120632041534245, 0.047761604001858, -0.120632041534245, -0.364874069642510, 0.921566702872299}
    };
    
    const mat b = {
        {0.488106762997528, 0.093547911260568, -0.485485431756243, -0.091984091707451, -0.000001945097416, 0.000009097619657 },
        {-0.093547911260568, 0.488106762997528, 0.091984091707451, -0.485485431756243, -0.000009097619657, -0.000001945097416 },
        {0.485485431756243, 0.091984091707451, -0.488106762997528, -0.093547911260568, 0.000001945097416, -0.000009097619657 },
        {-0.091984091707451, 0.485485431756243, 0.093547911260568, -0.488106762997528, 0.000009097619657, 0.000001945097416 },
        {0.038901948324797, -0.181952393142100, 0.038901948324797, -0.181952393142100, 0.000002613550852, 0.000001600210032 },
        {0.181952393142100, 0.038901948324797, 0.181952393142100, 0.038901948324797, -0.000001600210032, 0.000002613550852 }
    };

   // Action network initialization: 2*6*6*2 with tanh functions for hidden and output layers
    double num_hids1 = 6;
    double num_hids2 = 6;
    double num_outputs = 2;
    double num_inputs = use_idq ? 6: 4;

    load_matrix_from_csv(preloaded_idq_refs, "total_idq_refs.csv");
    
    const int numruns = 10;
    const int num_iterations = 1024;
    
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    
    colvec idq, dw, dw_y, rr;
    mat j_matrix, j_matrix2;
    rowvec e_hist_err, e_hist_err2;
    mat jj, ii, h_matrix, idq_his, idq_ref_his;
    double j_total, j_total_sum;
    double j_total2, j_total_sum2;
    vector<mat*> matrices;
    mat w1, w2, w3, w1_temp, w2_temp, w3_temp, starting_w1, starting_w2, starting_w3;
    if (use_shortcuts)
    {
        starting_w1 = mat(num_hids1, num_inputs + 1);
        starting_w2 = mat(num_hids2, num_hids1 + num_inputs + 1);
        starting_w3 = mat(num_outputs, num_hids2 + num_hids1 + num_inputs + 1);
    }
    else
    {
        starting_w1 = mat(num_hids1, num_inputs + 1);
        starting_w2 = mat(num_hids2, num_hids1 + 1);
        starting_w3 = mat(num_outputs, num_hids2 + 1);
    }

    mat w1_temp = mat(starting_w1.n_rows, starting_w1.n_cols);
    mat w2_temp = mat(starting_w2.n_rows, starting_w2.n_cols);
    mat w3_temp = mat(starting_w3.n_rows, starting_w3.n_cols);
    int num_weights = starting_w1.size() + starting_w2.size() + starting_w3.size();

    mat mu_mat = mat(1, 1);
    mat j_total_mat = mat(1, 1);

    bool p1, p2;

    const int num_samples = 10;
    const int num_parallel_jobs = 5;
    
    // Each worker needs to know which master to sends its results to / whether it is a master
    int mymaster = rank;
    while (rank % 10 != 0) mymaster++;
    int myjobindex = rank % 10;
    bool ismaster = myjobindex == 0;

    vector<ints> worker_pool, workers_in_other_pools;
    int startpoolid = (int)(rank/10) * 10;
    int endpoolid = startpoolid + 10;
    for (int i = startpoolid+1; i < endpoolid; i++)
    {
        worker_pool.push_back(i);
    }

    for (int i = 0; i < num_samples * num_parallel_jobs; i++)
    {
        if (i < startpoolid || i > endpoolid)
        {
            workers_in_other_pools.push_back(i);
        }
    }

    string filename("results" + itoa(rank) + ".csv");
    ofstream file_out;
    file_out.open(filename, std::ios_base::app);
    
    // Figure out how to have each process check at the top of this loop whether one process has found an answer
    MPI_Status status;
    bool answer_found = false;
    while (!answer_found)
    {
        // Each worker checks if there's an MPI_SOLUTION_FOUND message. If so, recv the message and quit
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_SOLUTION_FOUND, MPI_COMM_WORLD, &answer_found, &status);
        if (answer_found)
        {
            recv_matrices({&starting_w1, &starting_w2, &starting_w3}, MPI_ANY_SOURCE, MPI_SOLUTION_FOUND);
            break;
        }

        // Otherwise, keep searching for the right weights
        if (ismaster)
        {
            // Workers in worker pool need the same starting weights computed by master
            send_matrices({&starting_w1, &starting_w2, &starting_w3}, worker_pool);
        }
        else
        {
            recv_matrices({&starting_w1, &starting_w2, &starting_w3}, mymaster);
        }


        // FIXME: Make sure this produces two different references
        w1 = starting_w1.copy();
        w2 = starting_w2.copy();
        w3 = starting_w3.copy();
        mu = 1.0;

        for (int iteration = 1; iteration < num_iterations + 1; iteration++)
        {
            // Use FATT to calculate total cost of each trajectory, the error vector, and the jacobian matrix. 
            idq = idq_start_positions.col(myjobindex);
            unroll_trajectory_full(
                idq, vdq, myjobindex, trajectory_length,
                w3, w2, w1,
                use_idq, use_shortcuts,
                a, b,
                vmax, ts,
                gain1, gain2, gain3,
                cost_term_power,
                j_total, e_hist_err, j_matrix,
                idq_his, idq_ref_his
            );

            // Master will aggregate results from workers
            if (ismaster)
            {
                rowvec hist_err_total = rowvec(trajectory_length * num_samples, arma::fill::zeros);
                mat j_matrix_total = mat(trajectory_length * num_samples, num_weights, arma::fill::zeros);
                j_total_sum = 0;

                // The master first adds its piece to the aggregation variables
                j_total_sum += j_total;
                hist_err_total.cols(0, trajectory_length - 1) = e_hist_err;
                j_matrix_total.rows(0, trajectory_length - 1) = j_matrix;

                // Then, master waits to get chunks from the workers
                for (int i=1; i < num_samples; i++)
                {
                    MPI_Status status = recv_matrices({&j_total_mat, &e_hist_err, &j_matrix});

                    // Master needs to know which indices to use to update j_matrix_total / hist_err_total since these could arrive out of order
                    int senderid = status.MPI_Sender;
                    int senderidx = senderid % 10;    
                    int start = senderidx * trajectory_length;
                    int end = start + trajectory_length - 1;

                    // and update our totals
                    j_total_sum += j_total_mat[0];
                    hist_err_total.cols(start, end) = e_hist_err;
                    j_matrix_total.rows(start, end) = j_matrix;
                }
                
                // Now that we've computed j_total, hist_err_total, and j_matrix_total, 
                // master can do cholensky decomposition to solve for weight updates
                mat L = mat(num_weights, num_weights, arma::fill::zeros);
                mat R = mat(num_weights, num_weights, arma::fill::zeros);
                while (mu < mu_max)
                {
                    jj = j_matrix_total.t() * j_matrix_total;
                    ii = -1 * j_matrix_total.t() * arma::vectorise(hist_err_total);
                    h_matrix = jj + mu * arma::eye(num_weights, num_weights);

                    // cholensky decomposition to solve for weight updates
                    p1 = arma::chol(L, h_matrix, "lower");
                    p2 = arma::chol(R, h_matrix, "upper");

                    while (!p1 || !p2)
                    {
                        chol_counter++;
                        mu = mu * mu_inc;
                        if (mu == mu_max)
                        {
                            break;
                        }
                        h_matrix = jj + mu * arma::eye(num_weights, num_weights);
                        p1 = arma::chol(L, h_matrix, "lower");
                        p2 = arma::chol(R, h_matrix, "upper");
                    }
                    if (mu == mu_max)
                    {
                        break;
                    }
                    
                    arma::solve(dw_y, L, ii) && arma::solve(dw, R, dw_y);

                    w3_temp=w3+arma::reshape(dw.rows(0, w3.n_elem), w3.n_cols, w3.n_rows).t();
                    w2_temp=w2+arma::reshape(dw.rows(w3.n_elem, w3.n_elem + w2.n_elem), w2.n_cols, w2.n_rows).t();
                    w1_temp=w1+arma::reshape(dw.rows(w3.n_elem + w2.n_elem, dw.n_elem - 1), w1.n_cols, w1.n_rows).t();

                    // As soon as we've computed these new weights, we can give them to the workers to do their pieces
                    send_matrices({&w1_temp, &w2_temp, &w3_temp, &mu_mat, &j_total_mat}, worker_pool, MPI_NEW_JOB);

                    // Master needs to compute its piece of the sum
                    j_total_sum2 = 0;
   
                    idq = idq_start_positions.col(myjobindex);
                    unroll_trajectory_full(
                        idq, vdq, myjobindex, trajectory_length,
                        w3_temp, w2_temp, w1_temp,
                        use_idq, use_shortcuts,
                        a, b,
                        vmax, ts,
                        gain1, gain2, gain3,
                        cost_term_power,
                        j_total2, e_hist_err2, j_matrix2,
                        idq_his, idq_ref_his
                    );
    
                    j_total_sum2 += j_total2;

                    // Master needs to get sums from the workers
                    for (int i=1; i < num_samples; i++)
                    {
                        MPI_Recv(&j_total2, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_J_TOTAL, MPI_COMM_WORLD, NULL);
                        j_total_sum2 += j_total2;
                    }

                    if (j_total_sum2 < j_total_sum)
                    {
                        w3 = w3_temp;
                        w2 = w2_temp;
                        w1 = w1_temp;
                        //rr = join_cols(rr, colvec(j_total_sum2 / trajectory_length / num_samples));
                        mu = std::max(mu * mu_dec, mu_min);
                        file_out << setprecision(16)
                            << "iteration: " << iteration
                            << ", mu=" << mu
                            << ", J_total_sum=" << j_total_sum / trajectory_length / 10
                            << ", J_total_sum2=" << j_total_sum2 / trajectory_length / 10 << endl;
                        break;
                    }
                    mu = mu * mu_inc;
                } // while mu < mu_max loop ends here

                // Anytime we break out of this loop, we need to let our workers know
                mu_mat = mat({mu});
                j_total_mat = mat({j_total});
                send_matrices({&w1, &w2, &w3, &mu_mat, &j_total_mat}, workerids, MPI_LOOP_DONE);
            }

            // The workers have done their work, so they just send to the master and wait for the aggregation
            else
            {
                // Then we send the matrices
                send_matrices({&j_total_mat, &e_hist_err, &j_matrix}, mymaster, MPI_JOB_FINISHED);

                // Each worker will stay in this loop until master has decided we are done updating weights and mu
                while (true)
                {
                    MPI_Status status = recv_matrices({&w1_temp, &w2_temp, &w3_temp, &mu_mat, &j_total_mat}, mymaster);
                    if (status.MPI_TAG == MPI_NEW_JOB)
                    {
                        idq = idq_start_positions.col(myjobindex);
                        unroll_trajectory_full(
                            idq, vdq, myjobindex, trajectory_length,
                            w3_temp, w2_temp, w1_temp,
                            use_idq, use_shortcuts,
                            a, b,
                            vmax, ts,
                            gain1, gain2, gain3,
                            cost_term_power,
                            j_total2, e_hist_err2, j_matrix2,
                            idq_his, idq_ref_his
                        );
                        MPI_Send(&j_total2, 1, MPI_DOUBLE, 0, MPI_J_TOTAL, MPI_COMM_WORLD);
                    }
                    else if (status.MPI_TAG == MPI_LOOP_DONE)
                    {
                        w3 = w3_temp;
                        w2 = w2_temp;
                        w1 = w1_temp;
                        mu = mu_mat.at(0, 0);
                        j_total_sum2 = j_total_mat.at(0, 0);
                        break;
                    }
                }
            }

            // Each worker in the same pool can determine if they've found an answer
            if (j_total_sum2 / trajectory_length / num_samples < 0.1)
            {
                answer_found = true;
            }

            if (mu == mu_max && ismaster)
            {
                file_out << "reach mu_max " << endl;
            }

            // But we need to tell the workers in other pools that we've found the answer
            if (answer_found && ismaster)
            {
                file_out << "found solution, letting other workers know" << endl;
                send_matrices({&starting_w1, &starting_w2, &starting_w3}, workers_in_other_pools, MPI_SOLUTION_FOUND);
            }

            if (mu == mu_max || answer_found) break;

        } // iteration loop ends here
    } // while answer not found ends here
   
    // Only one total worker should be writing the final results 
    if (rank == 0)
    {
        save_matrix_to_csv(starting_w1, "starting_w1.csv");         
        save_matrix_to_csv(starting_w2, "starting_w2.csv");         
        save_matrix_to_csv(starting_w3, "starting_w3.csv");         
    }

    MPI_Finalize();
    return 0;
}

