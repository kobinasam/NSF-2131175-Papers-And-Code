

#include <cassert>
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

#include <armadillo>
#include <Eigen/Core>
#include <Eigen/Cholesky>

using namespace std;
using namespace arma;

using Eigen::MatrixXd;

typedef std::vector<double> stdvec;
typedef std::vector< std::vector<double> > stdvecvec;

#define _USE_MATH_DEFINES

#undef ARMA_OPENMP_THREADS
#define ARMA_OPENMP_THREADS 50 // Max threads used by armadillo for whatever logic they have parallelized. Cool
//#define ARMA_NO_DEBUG          // Only define this if we want to disable debugging for production code

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
double same_within_precision(const mat &m1, const mat &m2)
{
    double precision = 1e-16;
    while (!approx_equal(m1, m2, "both", precision, precision) and precision < 1)
    {
        precision = precision * 10;
    }
    return precision;
}

//// ==================================================================
//void vec_to_arr(const stdvec &V, double arr*) 
//{
//    int size=V.size();
//    for (int i=0; i < size; i++)
//    {
//        arr[i] = V[i];
//    }
//}
//

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

// ============================================================================
bool cholesky_armadillo(mat& L, mat& R, const mat &h_matrix)
{
    bool p1, p2, found;

    found = h_matrix.is_symmetric();
        
    double tolerance = 1e-16;
    while (!found and tolerance < 1e-1)
    {
        found = h_matrix.is_symmetric(tolerance);
        tolerance = tolerance * 10;
    }
    if (found)
    {
        cout << "Found symmetric matrix with tolerance: " << tolerance << endl;
    }
    else cout << "Did not find a symmetric matrix up to 1e-1 tolerance" << endl;

    tolerance = 1e-16;
    found = h_matrix.is_sympd();
    while (!found and tolerance < 1e-1)
    {
        found = h_matrix.is_sympd(tolerance);
        tolerance = tolerance * 10;
    }
    if (found)
    {
        cout << "Found symmetric positive definite matrix with tolerance: " << tolerance << endl;
    }
    else cout << "Did not find a symmetric positive definite matrix up to 1e-1 tolerance" << endl;


    mat eigval = mat(eig_sym(h_matrix));
    print_mat(eigval, "eigval");
    //std::cout << "============================================" << endl;
    //for (uword i = 0; i < eigval.n_elem; i++)
    //{
    //    std::cout << eigval.at(i) << endl;
    //}

    p1 = arma::chol(L, h_matrix, "lower");
    p2 = arma::chol(R, h_matrix, "upper");

    return p1 && p2;
}

// ============================================================================
bool cholesky_armadillo_perm(mat& L, mat& R, const mat &h_matrix)
{
    umat P_mat;
    bool p1 = arma::chol(L, P_mat, h_matrix, "lower");
    bool p2 = arma::chol(R, P_mat, h_matrix, "upper");
    return p1 && p2;
}

// ============================================================================
bool cholesky_eigen(mat& L, mat& R, mat& h_matrix)
{
    MatrixXd A(h_matrix.n_rows, h_matrix.n_cols);

    // FIXME: Probably inefficient, but do this for now
    for (int r = 0; r < h_matrix.n_rows; r++)
    {
        for (int c = 0; c < h_matrix.n_cols; c++)
        {
            A(r, c) = h_matrix.at(r, c); 
        }    
    }    

    Eigen::LLT<MatrixXd> lltofA(A);
    MatrixXd L_eigen = lltofA.matrixL();
    MatrixXd R_eigen = lltofA.matrixU();

    if (lltofA.info() == Eigen::Success)
    {
        // FIXME: Probably inefficient, but do this for now
        L.set_size(h_matrix.n_rows, h_matrix.n_cols);
        for (size_t i = 0, nRows = L_eigen.rows(), nCols = L_eigen.cols(); i < nCols; ++i)
        {
            for (size_t j = 0; j < nRows; ++j)
            {
                L.at(i, j) = L_eigen(i, j); 
            }
        }

        R.set_size(h_matrix.n_rows, h_matrix.n_cols);
        for (size_t i = 0, nRows = R_eigen.rows(), nCols = R_eigen.cols(); i < nCols; ++i)
        {
            for (size_t j = 0; j < nRows; ++j)
            {
                R.at(i, j) = R_eigen(i, j); 
            }
        }
        return true;
    }
    return false;
}


// ============================================================================
int main()
{

    // First four columns are the actual inputs. Last column is for the bias
    colvec test_input = {0.1, 0.2, 0.3, 0.4, -1};

    // Now lets set some weights to zero based on some condition. This is "dropout". 
    // Suppose we drop about 50% of the weights, which is 12 weights. I've sent 12 randomly to 0.0
    mat test_weights = {
        { 0.0, 0.027849821886705, 0.095716694824295, 0.0, 0.067873515485777 },
        { 0.090579193707562, 0.0, 0.048537564872284, 0.0, 0.075774013057833 },
        { 0.012698681629351, 0.095750683543430, 0.080028046888880, 0.0, 0.074313246812492 },
        { 0.091337585613902, 0.0, 0.014188633862722, 0.003571167857419, 0.0 },
        { 0.0, 0.015761308167755, 0.042176128262628, 0.0, 0.0 },
        { 0.009754040499941, 0.097059278176062, 0.0, 0.0, 0.017118668781156 }
    };

    // Output of first layer before applying activation function
    colvec sums = colvec(test_weights.n_rows);

    // Going to try both methods some large amount of times and measure the time
    int loops = 100000000;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Original method: Just do the matrix multiplication
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < loops; i++)
    {
        sums = test_weights * test_input;
    }
    end = std::chrono::steady_clock::now();
    cout << "Time for original method: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << endl;
    print_mat(sums, "Seeing sums");

    // New method: Do the matrix multplication but skip zeros?
    sums = colvec(test_weights.n_rows);
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < loops; i++)
    {
        for (int r = 0; r < test_weights.n_rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < test_weights.n_cols; c++)
            {
                if (test_weights(r, c) != 0.0)
                {
                    sum += test_weights(r, c) * test_input(c);
                }
            }
            sums(r) = sum;
        }
    }
    end = std::chrono::steady_clock::now();
    cout << "Time for new method: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << endl;
    print_mat(sums, "Seeing sums");

    //test_weights.reshape(1, test_weights.n_elem);

    mat trend;

    rowvec combined_weights = rowvec(test_weights.n_elem * 2);
    combined_weights.cols(0, test_weights.n_elem - 1) = arma::reshape(test_weights, 1, test_weights.n_elem);
    combined_weights.cols(test_weights.n_elem, test_weights.n_elem * 2 - 1) = arma::reshape(test_weights, 1, test_weights.n_elem);

    trend.insert_rows(trend.n_rows, combined_weights);
    trend.insert_rows(trend.n_rows, combined_weights);

    save_matrix_to_csv(trend, "testweights.csv");

}

