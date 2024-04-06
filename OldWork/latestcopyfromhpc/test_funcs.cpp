

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
void vec_to_arr(const stdvec &V, double arr*) 
{
    int size=V.size();
    for (int i=0; i < size; i++)
    {
        arr[i] = V[i];
    }
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
void mats_to_std_array(vector<mat*> Ms, double *arr)
{
    // Combines all matrices in Ms into a single flattened array
    int arr_idx = 0; 
    for (int i=0; i < Ms.size(); i++)
    {
        // Use pointer arithmetic to increment pointer to fill the right spot in the buffer
        mat_to_std_array(Ms[i], arr + arr_idx);
        arr_idx = Ms[i]->n_elems;
    }
}

// ==================================================================
void std_array_to_mat(double* arr, mat &M)
{
    M = mat(arr, 
}

// ==================================================================
void std_array_to_mats(double* arr, vector<mat*> Ms)
{
    // Fill matrices with values from array
    int arr_idx = 0; 
    for (int i=0; i < Ms.size(); i++)
    {
        // Use pointer arithmetic to increment pointer to fill the right spot in the buffer
        mat_to_std_array(Ms[i], arr + arr_idx);
        arr_idx = Ms[i]->n_elems;
    }
}

// ==================================================================
int count_in_matrices(vector<mat*> matrices)
{
    int total_count=0;
    for (int i=0; i < matrices.size(); i++)
    {
        int count = matrices[i]->n_elems;
        total_count += count;
    }

    return total_count;
}

// ==================================================================
int cholesky(vector<mat*> matrices)
{
    int total_count=0;
    for (int i=0; i < matrices.size(); i++)
    {
        int count = matrices[i]->n_elems;
        total_count += count;
    }

    return total_count;
}


// ============================================================================
int main()
{

    count_in_matrices(vector<mat*> matrices)
    
}
