

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
    mat m1;
    mat m2;
    mat m3;
    vector<mat*> matrices;
    double *arr;
    double *arr1;
    stdvecvec vecvec;

    // ============================================================================
    // count_in_matrices test cases
    // ============================================================================
 
    // Case: Empty vector of matrices
    matrices.clear();
    assert (count_in_matrices({}) == 0);

    // Case: Vector with one empty matrix
    m1 = mat(0, 0);
    assert (count_in_matrices({&m1}) == 0);

    // Case: vector with one matrix with values
    m1 = mat(1, 3);
    assert (count_in_matrices({&m1}) == 3);

    // Case: vector with multiple matrices with values
    m1 = mat(1, 3);
    m2 = mat(1, 3);
    assert (count_in_matrices({&m1, &m2}) == 6);

    // ============================================================================
    // mat_to_std_vec
    // ============================================================================

    // Case: matrix with one column and one row
    m1 = mat(1, 1, arma::fill::ones);
    vecvec = mat_to_std_vec(m1);
    assert(vecvec.size() == 1);
    assert(vecvec[0].size() == 1);
    assert(vecvec[0][0] == 1);

    // Case: matrix with mult cols and rows
    m1 = mat(2, 2);
    m1.at(0, 0) = 1;
    m1.at(0, 1) = 2;
    m1.at(1, 0) = 3;
    m1.at(1, 1) = 4;
    vecvec = mat_to_std_vec(m1);
    assert(vecvec.size() == 2);
    assert(vecvec[0].size() == 2);
    assert(vecvec[0][0] == 1);
    assert(vecvec[0][1] == 2);
    assert(vecvec[1][0] == 3);
    assert(vecvec[1][1] == 4);

    // ============================================================================
    // mat_to_std_array test cases
    // ============================================================================

    arr = new double[10];
    m1 = mat(2, 5, arma::fill::ones);
    mat_to_std_array(&m1, arr);

    for (int i = 0; i < 10; i++)
    {
        assert(arr[i] == 1);
    }
    delete[] arr;

    // ============================================================================
    // std_array_to_mat test cases
    // ============================================================================
    arr = new double[6];
    m1 = mat(2, 3);
    arr[0] = 0;
    arr[1] = 1;
    arr[2] = 2;
    arr[3] = 3;
    arr[4] = 4;
    arr[5] = 5;

    std_array_to_mat(arr, m1);

    assert(m1.at(0, 0) == 0);
    assert(m1.at(0, 1) == 1);
    assert(m1.at(0, 2) == 2);
    assert(m1.at(1, 0) == 3);
    assert(m1.at(1, 1) == 4);
    assert(m1.at(1, 2) == 5);
    delete[] arr;


    // ============================================================================
    // mat_to_std_array composition
    // ============================================================================

    arr = new double[6];
    arr1 = new double[6];
    m1 = mat(2, 3, arma::fill::randu);
    m2 = mat(2, 3, arma::fill::ones);

    mat_to_std_array(&m1, arr);
    std_array_to_mat(arr, m2);
    assert (arma::approx_equal(m1, m2, "absdiff", 0.00000001));

    m1 = mat(2, 3, arma::fill::zeros);
    for (int i = 0; i < 6; i++)
    {
        arr[i] = i;
    }

    std_array_to_mat(arr, m1);
    mat_to_std_array(&m1, arr1);

    assert(arr1[0] == 0);
    assert(arr1[1] == 1);
    assert(arr1[2] == 2);
    assert(arr1[3] == 3);
    assert(arr1[4] == 4);
    assert(arr1[5] == 5);
    delete[] arr;
    delete[] arr1;

    // ============================================================================
    // mats_to_std_array test cases
    // ============================================================================

    // Case: matrix with multiple columns / rows to multidimensional array
    arr = new double[6];
    m1 = mat(1, 2, arma::fill::ones) * 1;
    m2 = mat(1, 1, arma::fill::ones) * 2;
    m3 = mat(1, 3, arma::fill::ones) * 3;
    matrices.clear();
    matrices.push_back(&m1);
    matrices.push_back(&m2);
    matrices.push_back(&m3);
    mats_to_std_array(matrices, arr);

    assert(arr[0] == 1);
    assert(arr[1] == 1);
    assert(arr[2] == 2);
    assert(arr[3] == 3);
    assert(arr[4] == 3);
    assert(arr[5] == 3);

    delete[] arr;

    // ============================================================================
    // std_array_to_mats test cases
    // ============================================================================
 
    // Case: matrix with multiple columns / rows to multidimensional array
    arr = new double[6];
    for (int i = 0; i < 6; i++)
    {
        arr[i] = i;
    }

    m1 = mat(1, 2);
    m2 = mat(1, 1);
    m3 = mat(3, 1);

    matrices.clear();
    matrices.push_back(&m1);
    matrices.push_back(&m2);
    matrices.push_back(&m3);
    std_array_to_mats(arr, matrices);

    assert(matrices[0]->at(0, 0) == 0);
    assert(matrices[0]->at(0, 1) == 1);
    assert(matrices[1]->at(0, 0) == 2);
    assert(matrices[2]->at(0, 0) == 3);
    assert(matrices[2]->at(1, 0) == 4);
    assert(matrices[2]->at(2, 0) == 5);

    delete[] arr;

    // Case: matrix with multiple columns / rows to multidimensional array
    arr = new double[6];
    for (int i = 0; i < 6; i++)
    {
        arr[i] = i;
    }

    m1 = mat(1, 2);
    m2 = mat(1, 1);
    m3 = mat(3, 1);

    matrices.clear();
    matrices.push_back(&m1);
    matrices.push_back(&m2);
    matrices.push_back(&m3);
    std_array_to_mats(arr, matrices);

    assert(matrices[0]->at(0, 0) == 0);
    assert(matrices[0]->at(0, 1) == 1);
    assert(matrices[1]->at(0, 0) == 2);
    assert(matrices[2]->at(0, 0) == 3);
    assert(matrices[2]->at(1, 0) == 4);
    assert(matrices[2]->at(2, 0) == 5);

    delete[] arr;

    // ============================================================================
    // cholesky test case
    // ============================================================================

    mat h_matrix, expected_h_matrix, expected_L, expected_R, L, R;
    load_matrix_from_csv(h_matrix, "h_matrix.csv");
    load_matrix_from_csv(expected_L, "l.csv");
    load_matrix_from_csv(expected_R, "r.csv");

    //bool success = cholesky_armadillo(L, R, h_matrix);
    //bool success = cholesky_armadillo_perm(L, R, h_matrix);
    bool success = cholesky_eigen(L, R, h_matrix);
    cout << "Success? " << success << endl;

    assert(success);
    
    mat h_matrix_from_chol_factor = L * L.t();
    cout << "A = L * L.t()? " << same_within_precision(h_matrix, h_matrix_from_chol_factor) << endl;

    //success = cholesky_eigen();

    cout << "L: " << same_within_precision(L, expected_L) << endl;
    cout << "R: " << same_within_precision(R, expected_R) << endl;

    print_mat(L.col(0).rows(0, 4), "Armadillo L");
    print_mat(expected_L.col(0).rows(0, 4), "Matlab L");

    print_mat(R.col(R.n_cols-1).rows(0, 4), "Armadillo R");
    print_mat(expected_R.col(expected_R.n_cols-1).rows(0, 4), "Matlab R");

    mat h_matrix_armadillo;
    load_matrix_from_csv(h_matrix_armadillo, "h_matrix_armadillo.csv");

    success = cholesky_eigen(h_matrix_armadillo, L, R);
    cout << "Success on armdillo matrix? " << success << endl;

    //// Print the precision to which the matrices are the same (1 means they aren't at all the same)
    //cout << "L: " << same_within_precision(L, expected_L) << endl;
    //cout << "R: " << same_within_precision(R, expected_R) << endl;

    //cout << "Shape of L: " << L.n_cols << ", " << L.n_rows << endl;
    //// Show first five digits of matlab L/R and armadillo L/R

    //assert(approx_equal(L, expected_L, "both", 1e-16, 1e-16));
    //assert(approx_equal(R, expected_R, "both", 1e-16, 1e-16));

    //mat A, expected_L2, expected_R2, L2, R2;
    //load_matrix_from_csv(A, "A_chol.csv");
    //load_matrix_from_csv(expected_L2, "L_chol.csv");
    //load_matrix_from_csv(expected_R2, "R_chol.csv");

    //p1 = arma::chol(L2, A, "lower");
    //p2 = arma::chol(R2, A, "upper");

    //// Should be a success for both
    //assert(p1);
    //assert(p2);

    //cout << "All tests passed!" << endl;
 }

