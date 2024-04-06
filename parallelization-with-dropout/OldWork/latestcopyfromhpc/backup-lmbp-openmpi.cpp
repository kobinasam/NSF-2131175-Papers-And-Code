
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

using namespace std;
using namespace arma;

#define _USE_MATH_DEFINES

#undef ARMA_OPENMP_THREADS
#define ARMA_OPENMP_THREADS 50 // Max threads used by armadillo for whatever logic they have parallelized. Cool
#define ARMA_NO_DEBUG          // Only define this if we want to disable debugging for production code

// Used to distingish message types in sending/receiving messages
#define MPI_PAYLOAD_TAG 0
#define MPI_J_TOTAL_TAG 1

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
// Floating point math sucks. Use this to guarantee you can round up
// or down on a floating point value
float round_to_decimal(float var, int dec)
{
    int multiplier = pow(10, dec);
    int value = (int)(var * multiplier + .5);
    return (float)value / multiplier;
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
//void vec_to_arr(const stdvec &V, double arr*) 
//{
//    int size=V.size();
//    for (int i=0; i < size; i++)
//    {
//        arr[i] = V[i];
//    }
//}

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
void send_matrices(vector<mat*> matrices, int receiver)
{
    // We don't want to make multiple calls to send multiple matrices, so here's what we do
    // The caller just packages the pointers to the matrices into a vector
    // Then, this function iterates over those matrices to figure out how the total count of the data to send
    // Then we can know how to call one MPI_Send() rather than N sends for N matrices

    int total_count=count_in_matrices(matrices);
    double payload = new double[total_count];
    mats_to_std_array(matrices, payload);
    MPI_Send(payload, total_count, MPI_DOUBLE, receiver, MPI_PAYLOAD_TAG, MPI_COMM_WORLD);

    delete payload[];
}

// ==================================================================
void recv_matrices(vector<mat*> matrices, int sender)
{
    // Like with send_matrices, the idea is to provide a vector of matrices
    // (the size of each matrix determined by the caller) and then we just fill those matrices
    // with the values from the sender
    int total_count=count_in_matrices(matrices);
    double payload = new double[total_count];
    MPI_Recv(payload, total_count, MPI_DOUBLE, sender, MPI_PAYLOAD_TAG, MPI_COMM_WORLD);

    std_array_to_mats(payload, matrices)

    delete payload[];
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
int main()
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
   
    if (rank == 0)
    {
        string filename("results.csv");
        ofstream file_out;
        file_out.open(filename, std::ios_base::app);
    } 

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
    
    const mat idq_start_positions_temp = {
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

    const mat starting_w1 = {
        { 0.081472368639318, 0.027849821886705, 0.095716694824295, 0.079220732955955, 0.067873515485777 },
        { 0.090579193707562, 0.054688151920498, 0.048537564872284, 0.095949242639290, 0.075774013057833 },
        { 0.012698681629351, 0.095750683543430, 0.080028046888880, 0.065574069915659, 0.074313246812492 },
        { 0.091337585613902, 0.096488853519928, 0.014188633862722, 0.003571167857419, 0.039222701953417 },
        { 0.063235924622541, 0.015761308167755, 0.042176128262628, 0.084912930586878, 0.065547789017756 },
        { 0.009754040499941, 0.097059278176062, 0.091573552518907, 0.093399324775755, 0.017118668781156 }
    };

    const mat starting_w2 = {
        { 0.012649986532930, 0.031747977514944, 0.055573794271939, 0.055778896675488, 0.025779225057201, 0.040218398522248, 0.087111112191539 },
        { 0.013430330431357, 0.031642899914629, 0.018443366775765, 0.031342898993659, 0.039679931863314, 0.062067194719958, 0.035077674488589 },
        { 0.009859409271100, 0.021756330942282, 0.021203084253232, 0.016620356290215, 0.007399476957694, 0.015436980547927, 0.068553570874754 },
        { 0.014202724843193, 0.025104184601574, 0.007734680811268, 0.062249725927990, 0.068409606696201, 0.038134520444447, 0.029414863376785 },
        { 0.016825129849153, 0.089292240528598, 0.091380041077957, 0.098793473495250, 0.040238833269616, 0.016113397184936, 0.053062930385689 },
        { 0.019624892225696, 0.070322322455629, 0.070671521769693, 0.017043202305688, 0.098283520139395, 0.075811243132742, 0.083242338628518 }
    };

    const mat starting_w3 = {
        { 0.002053577465818, 0.065369988900825, 0.016351236852753, 0.079465788538875, 0.044003559576025, 0.075194639386745, 0.006418708739190 },
        { 0.092367561262041, 0.093261357204856, 0.092109725589220, 0.057739419670665, 0.025761373671244, 0.022866948210550, 0.076732951077657 }
    };

    const int num_weights = starting_w1.size() + starting_w2.size() + starting_w3.size();
    
    load_matrix_from_csv(preloaded_idq_refs, "total_idq_refs.csv");
    
    const int numruns = 10;
    const int num_iterations = 1024;
    int num_samples;
    
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    
    mat idq_start_positions;
    colvec idq, dw, dw_y, rr;
    mat w1_temp, w2_temp, w3_temp;
    mat j_matrix, j_matrix2;
    rowvec e_hist_err, e_hist_err2;
    mat jj, ii, h_matrix;
    double j_total, j_total_subsum, j_total_sum;
    double j_total2, j_total_subsum2, j_total_sum2;
    vector<mat*> matrices;

    bool p1, p2;

    for (int num_sample_time = 1; num_sample_time <=10; num_sample_time++)
    {
        if (rank == 0)
        {
            begin = std::chrono::steady_clock::now();
        }

        num_samples = 10 * num_sample_time;

        // We have no need for more workers than we have samples, so use active_workers
        int active_workers = min(num_samples, total_workers);

        // each worker computes how many elements it needs to process    
        int samples_to_process = num_elems_to_process(rank, num_samples, active_workers);
 
        // Each worker computes its reserved indices for depositing values into the aggregation variables
        pair<int, int> range_of_samples = end_start_elems_to_process(rank, num_samples, active_workers);

        // The master needs to know which samples are handled by each worker
        int rolling_sum = 0;
        map<int, int> starting_indices;
        for (int i = 0; i < active_workers-1; i++)
        {
            rolling_sum += num_elems_to_process(i, num_samples, active_workers);
            starting_indices[i+1] = rolling_sum;
        }
        
        // some workers may have nothing to do (if workerid >= num_samples)
        if (samples_to_process > 0)
        {
            idq_start_positions.clear();
            for (int istart=0; istart < num_sample_time; istart++)
            {
                idq_start_positions = join_horiz(idq_start_positions, idq_start_positions_temp);
            }
    
            for (int runi=0; runi < numruns; runi++)
            {
                mat w1 = starting_w1;
                mat w2 = starting_w2;
                mat w3 = starting_w3;

                mu = 1.0;
                for (int iteration = 1; iteration < num_iterations + 1; iteration++)
                {
                    // Use FATT to calculate total cost of each trajectory, the error vector, and the jacobian matrix. 

                    mat idq_his, idq_ref_his;

                    rowvec hist_err_total = rowvec(trajectory_length * num_samples, arma::fill::zeros);
                    mat j_matrix_total = mat(trajectory_length * num_samples, num_weights, arma::fill::zeros);
                    j_total_sum = 0;

                    rowvec hist_err_subtotal = rowvec(trajectory_length * samples_to_process, arma::fill::zeros);
                    mat j_matrix_subtotal = mat(trajectory_length * samples_to_process, num_weights, arma::fill::zeros);
                    j_total_subsum = 0;
                   
                    //cout << "Samples to process: " << samples_to_process << endl; 
                    //cout << "hist_err_subtotal: (" << hist_err_subtotal.n_rows << ", " << hist_err_subtotal.n_cols << ")" << endl;
                    //cout << "j_Matrix_subtotal: (" << j_matrix_subtotal.n_rows << ", " << j_matrix_subtotal.n_cols << ")" << endl;

                    // Each worker (master included) does its own part for this loop
                    for (int i = range_of_samples.first, j = 0; i < range_of_samples.second; i++, j++)
                    {
                        idq = idq_start_positions.col(i);
                        unroll_trajectory_full(
                            idq, vdq, i, trajectory_length,
                            w3, w2, w1,
                            use_idq, use_shortcuts,
                            a, b,
                            vmax, ts,
                            gain1, gain2, gain3,
                            cost_term_power,
                            j_total, e_hist_err, j_matrix,
                            idq_his, idq_ref_his
                        );

                        if (i < 10)
                        {
                            //cout << "e_hist_err: (" << e_hist_err.n_rows << ", " << e_hist_err.n_cols << ")" << endl;
                            //cout << "j_matrix: (" << j_matrix.n_rows << ", " << j_matrix.n_cols << ")" << endl;

                            j_total_subsum += j_total;
                            hist_err_subtotal.cols(j * trajectory_length, (j+1) * (trajectory_length) - 1) = e_hist_err;
                            j_matrix_subtotal.rows(j * trajectory_length, (j+1) * (trajectory_length) - 1) = j_matrix;
                        }
                    }
                    //cout << "Process " << rank << " computed j_subsum: " << j_total_subsum << endl;
                    //print_mat(hist_err_subtotal.cols(0, 4).t(), "hist_error_subtotal");

                    // Master will aggregate results from workers
                    if (rank == 0)
                    {
                        // The master first adds its piece to the aggregation variables, then waits to get chunks from the workers
                        j_total_sum += j_total_subsum;
                        hist_err_total.cols(0, samples_to_process * trajectory_length - 1) = hist_err_subtotal;
                        j_matrix_total.rows(0, samples_to_process * trajectory_length - 1) = j_matrix_subtotal;

                        MPI_Status status;
                        for (int i=1; i < active_workers; i++)
                        {
                            // The first thing any worker sends is the j_total, which is a single value
                            MPI_Recv(&j_total_subsum, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_J_TOTAL_TAG, MPI_COMM_WORLD, &status);
                            //cout << "Total subsum received from worker " << status.MPI_SOURCE << ": " << j_total_subsum << endl;    

                            j_total_sum += j_total_subsum;

                            // That same worker that sent the j_total will immediately send the arrays
                            // We can know how many elements they intend to send since we know their id
                            int senderid = status.MPI_SOURCE;
                            int sender_samples = num_elems_to_process(senderid, num_samples, active_workers);

                            // we receive these subtotals from sender
                            hist_err_subtotal = rowvec(trajectory_length * sender_samples);
                            j_matrix_subtotal = mat(trajectory_length * sender_samples, num_weights);
                            recv_matrices({&hist_err_subtotal, &j_matrix_subtotal}, senderid);

                            int start = starting_indices[senderid] * trajectory_length;
                            int end = start + (sender_samples * trajectory_length) - 1;

                            // and update our totals
                            hist_err_total.cols(start, end) = hist_err_subtotal;
                            j_matrix_total.rows(start, end) = j_matrix_subtotal;

                        }

                        // The workers have informed master of their computations so that master can compute the aggregations,
                        // but the workers also need to be informed of the final aggregations to continue to do their work
                        for (int i=1; i < active_workers; i++)
                        {
                            // So we send each worker the totals
                            send_matrices({&hist_err_total, &j_matrix_total}, i);
                        }
                    }
                        
                    // The workers have done their work, so they just send to the master and wait for the aggregation
                    else
                    {
                        // We send the subsum first so that the master can know which worker finished first
                        MPI_Send(&j_total_subsum, 1, MPI_DOUBLE, 0, MPI_J_TOTAL_TAG, MPI_COMM_WORLD);

                        // Then we send the matrices
                        send_matrices({&hist_err_subtotal, &j_matrix_subtotal}, 0);

                        // We wait for the master node to tell us what the totals are so we can continue
                        recv_matrices({&hist_err_total, j_matrix_total}, senderid);
                    }

                    // Now that we've computed j_total, hist_err_total, and j_matrix_total, 
                    // we do cholensky decomposition to solve for weight updates
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

                        // Then we need to split up the work amongst the workers...
                        j_total_sum2 = 0;
                        j_total_subsum2 = 0;
   
                        // Each worker (master included) does its own part for this loop
                        for (int i = range_of_samples.first; i < range_of_samples.second; i++)
                        {
                            idq = idq_start_positions.col(i);
                            unroll_trajectory_full(
                                idq, vdq, i, trajectory_length,
                                w3_temp, w2_temp, w1_temp,
                                use_idq, use_shortcuts,
                                a, b,
                                vmax, ts,
                                gain1, gain2, gain3,
                                cost_term_power,
                                j_total2, e_hist_err2, j_matrix2,
                                idq_his, idq_ref_his
                            );
    
                            if (i < 10)
                            {
                                j_total_subsum2 += j_total2;
                            }
                        }

                        // Master will aggregate results from workers
                        if (rank == 0)
                        {
                            // The master first adds its piece to the aggregation variables, then waits to get chunks from the workers
                            j_total_sum2 += j_total_subsum2;

                            // Collects subsums from workers
                            for (int i=1; i < active_workers; i++)
                            {
                                MPI_Recv(&j_total_subsum2, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_J_TOTAL_TAG, MPI_COMM_WORLD, NULL);
                                j_total_sum2 += j_total_subsum2;
                            }

                            // Sends final result back to workers
                            for (int i=1; i < active_workers; i++)
                            {
                                MPI_Send(&j_total_sum2, 1, MPI_DOUBLE, i, MPI_J_TOTAL_TAG, MPI_COMM_WORLD, NULL);
                            }

                        }

                        // The workers have done their work, so they just send to the master and wait for final sum
                        else
                        {
                            MPI_Send(&j_total_subsum2, 1, MPI_DOUBLE, 0, MPI_J_TOTAL_TAG, MPI_COMM_WORLD);
                            MPI_Recv(&j_total_sum2, 1, MPI_DOUBLE, 0, MPI_J_TOTAL_TAG, MPI_COMM_WORLD);
                        }

                        // FIXME: Force this c++ implementation to run as many iterations as matlab for performance comparison
                        int matlab_iterations = 62;
                        if (j_total_sum2 < j_total_sum || (iteration <= matlab_iterations && mu * mu_inc == mu_max))
                        //if (j_total_sum2 < j_total_sum)
                        {
                            w3 = w3_temp;
                            w2 = w2_temp;
                            w1 = w1_temp;
                            rr = join_cols(rr, colvec(j_total_sum2 / trajectory_length / num_samples));
                            mu = std::max(mu * mu_dec, mu_min);
                            if (rank == 0)
                            {
                                std::cout << setprecision(16)
                                    << "iteration: " << iteration
                                    << ", mu=" << mu
                                    << ", J_total_sum2=" << j_total_sum2 / trajectory_length / 10 << endl;
                            }

                            if (iteration != matlab_iterations)
                            {
                                    break;
                            }
                        }
    
                        mu = mu * mu_inc;
                        if (iteration == matlab_iterations)
                        {
                            // Force the run to finish at max iterations
                            mu = mu_max;
                        }

                    } // while mu < mu_max loop ends here

                    if (mu == mu_max)
                    {
                        // FIXME: Only master should cout this
                        std::cout << "reach mu_max " << endl;
                        break;
                    }

                } // iteration loop ends here

            } // run loop ends here
            if (rank == 0)
            {
                end = std::chrono::steady_clock::now();
                file_out << total_workers << ", " << num_samples << ", " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << endl;
            }
        } // samples_to_process > 0 if-then

    } // num_sample_times loop ends here
    if (rank == 0)
    {
        file_out.close();
    }
    MPI_Finalize();
    return 0;
}
