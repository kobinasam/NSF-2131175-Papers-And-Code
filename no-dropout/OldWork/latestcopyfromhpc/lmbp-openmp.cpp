
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
#include <vector>

#include <omp.h>

#include <armadillo>

using namespace std;
using namespace arma;

#define _USE_MATH_DEFINES

//#undef ARMA_OPENMP_THREADS
//#define ARMA_OPENMP_THREADS 50 // Max threads used by armadillo for whatever logic they have parallelized. Cool
//#define ARMA_NO_DEBUG          // Only define this if we want to disable debugging for production code


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

// ======================================================================
// Implements calculateIdq_ref.m
colvec calculate_idq_ref(
    const int &trajectory_number,
    const int &time_step,
    const double &ts, 
    const mat &preloaded_idq_refs)
{
    colvec idq_ref;
    // NOTE: trajectory_number and time_step come from zero-indexed, rather than one-indexed code
    // So when comparing to matlab, we must add an extra value
    //float period = 0.1;
	//int change_number = floor(round_to_decimal(time_step * ts / period, 2)) + 1;

    //// Well the seed computation is inconsistent.
    //int seed = ((trajectory_number+1) % 10) * 10000 + change_number;
    //srand(seed);
    //arma_rng::set_seed(seed);
    //idq_ref = colvec({ -300, -300 }) + colvec({ 600, 360 }) % mat(2, 1, arma::fill::randu);
    //idq_ref = arma::round(idq_ref * 10) / 10;

    // The calculation for the seed is all fucked, so just use the values from matlab
    //
    // REMINDER: Look into whether this shit is fucked
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
    const mat &preloaded_idq_refs,
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

    mat didq_dw, didq_dw_matrix_sum;
    didq_dw = mat(6, num_weights, arma::fill::zeros);
    const mat dvdq_dw = mat(2, num_weights, arma::fill::zeros);
    j_matrix = mat(trajectory_length+1, num_weights, arma::fill::zeros);
    didq_dw_matrix_sum = mat(2, num_weights, arma::fill::zeros);

    mat err_integral, dudq_dw;
    colvec idq_ref, idq_refi;

    // outputs of net_action
    mat o3, udq, ndq, dnet_dw1, dnet_didq1, dnet_dhist_err1, dnet_dw2, dnet_didq2, dnet_dhist_err2;

    for (int len = 0; len < trajectory_length; len++)
    {
        err_integral = ts * (arma::sum(hist_err, 1) - hist_err.col(len) / 2.0);
        idq_ref = calculate_idq_ref(trajectory_number, len, ts, preloaded_idq_refs);
        //cout << "num: " << trajectory_number << ", " << len << ": " << idq_ref[0] << ", " << idq_ref[1] << endl;
        idq_ref_his.col(len) = idq_ref;

		hist_err.col(len) = (idq.rows(0, 1) - idq_ref); // when the error is too small, the calculation becomes inaccurate
		e_hist_err.col(len) = pow((arma::sum(hist_err.col(len) % hist_err.col(len))), cost_term_power / 2.0);// the calculation process of pow is slightly from that in Matlab

		net_action(idq, idq_ref, err_integral, w1, w2, w3, false, use_shortcuts, use_idq, gain1, gain2, gain3, ndq, dnet_dw1, dnet_didq1, dnet_dhist_err1);
		udq = ndq * vmax;

        // REMINDER: All three of these lines are causing a race condition
        mat tmp, tmp1, tmp2, tmp3;
        #pragma omp critical
        {
		    net_action(idq, idq_ref, err_integral, w1, w2, w3, true, use_shortcuts, use_idq, gain1, gain2, gain3, o3, dnet_dw2, dnet_didq2, dnet_dhist_err2);
            if (use_idq)
            {
                dudq_dw = vmax * (dnet_dw2 + dnet_didq2 * didq_dw + dnet_dhist_err2 * ts * didq_dw_matrix_sum);
            }
            else
            {
                dudq_dw = vmax * (dnet_dw2 + dnet_didq2 * didq_dw.rows(0, 1) + dnet_dhist_err2 * ts * (didq_dw_matrix_sum - didq_dw.rows(0, 1) / 2.0));
            }

            tmp = join_vert(dvdq_dw, dudq_dw, dvdq_dw);
        }
        tmp3 = mat(b * tmp);
        tmp = tmp3;    
        tmp1 = a * didq_dw;
        didq_dw = tmp + tmp1;
        tmp2 = didq_dw.rows(0, 1);     
        didq_dw_matrix_sum = didq_dw_matrix_sum + tmp2;
 
        //int id = omp_get_thread_num();
        //if (id == 0)
        //{
        //    cout << trajectory_number << ", " << len << endl;
        //    print_mat(dnet_dw2, "dnet_dw2");
        //    print_mat(dnet_didq2, "dnet_didq2");
        //    print_mat(dnet_dhist_err2, "dnet_dhist_err2");
        //}

        //#pragma omp critical
        //{
        //    didq_dw = a * didq_dw + b * join_vert(dvdq_dw, dudq_dw, dvdq_dw);
        //}
        
		//net_action(idq, idq_ref, err_integral, w1, w2, w3, true, use_shortcuts, use_idq, gain1, gain2, gain3, o3, dnet_dw2, dnet_didq2, dnet_dhist_err2);

        //if (use_idq)
        //{
        //    dudq_dw = vmax * (dnet_dw2 + dnet_didq2 * didq_dw + dnet_dhist_err2 * ts * didq_dw_matrix_sum);
        //}
        //else
        //{
        //    dudq_dw = vmax * (dnet_dw2 + dnet_didq2 * didq_dw.rows(0, 1) + dnet_dhist_err2 * ts * (didq_dw_matrix_sum - didq_dw.rows(0, 1) / 2.0));
        //}
        
        //#pragma omp critical
        //{
        //mat tmp = join_vert(dvdq_dw, dudq_dw, dvdq_dw);

        //if (trajectory_number == 0 && len == 999)
        //{
        //    mat expected_tmp = {
        //        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //        {-80.03538925343648, -39.59267237640256, -57.03463952937736, -34.90967316222657, -64.95213511095491, -87.66412044013009, -733.9833534367694, 1.920928026288252, 0.9741243263114676, 1.352064020576435, 0.8848576856190868, 1.626108796731446, 2.127532916903472, 17.04226644913594, 0.04295081425188142, 0.0500746625483858, 0.0169359703242894, 0.04812768560778539, 0.03573711293098041, 0.01316059404634312, 0.06465392467378824, -1.170639177787392, -5.343426538457732, -6.859345363533037, -2.843054497201283, -3.745144329147517, -5.170235009922404, -46.25679094124386, -0.2303094356969518, -1.164697376625196, -1.531925691000366, -0.6029972968739286, -0.8158252921442464, -1.154382901724245, -10.36921639293737, -1.473985248303774, -6.63597998682328, -8.488717973869525, -3.544332204555287, -4.651483401653051, -6.398624426539862, -57.21340579303737, -0.8173075802570902, -3.66937771059614, -4.69048967693022, -1.961368539830464, -2.572091744274443, -3.535624222159202, -31.61008594856025, -1.406265221983757, -6.280670721525944, -8.017611793361189, -3.362086167347855, -4.402653867116375, -6.043652519366678, -54.02077841864765, 2.183873193471465, -3.125735667984567, 3.021339278353081, -3.126502537586687, -3.125670002015284, 6.702365748481228, -9.615807286419052, 9.29392265140136, -9.617836893102915, -9.615636173258824, 5.617529367313502, -8.037596497144081, 7.769133563797006, -8.039556752599125, -8.037424308060571, 6.425778019730418, -9.190628261804161, 8.884059692852199, -9.193073046117826, -9.190418535241729, 8.580139568255035, -12.32482486456225, 11.91140233984417, -12.32700737601056, -12.32464581444046, 6.772209396315009, -9.692572378876362, 9.368698929499926, -9.694874856660173, -9.692373556879151},
        //        {0.004397903925990906, -0.008182389399534319, 0.01045572495517951, -0.01888498613445585, -0.02650714335381476, -0.005591070575899986, 0.2901974625890027, -77.23779109385518, -38.17434983209033, -55.06528067807105, -33.6213395515508, -62.58408695437922, -84.56579661255608, -709.155118692816, -1.638345109777669, -7.475933425343838, -9.598526388640542, -3.977649964293101, -5.240091330179294, -7.234992968092472, -64.72545245188616, -1.680702204183678, -7.627892276028746, -9.778383076356867, -4.064876978514759, -5.346575214483623, -7.370602793497551, -65.92616867148453, -1.645852957245445, -7.501408392067194, -9.627983478409817, -3.99256110750887, -5.257939138125001, -7.25720454683288, -64.92152714488269, -1.048428878349308, -4.732610671724417, -6.057323186862409, -2.525977404305515, -3.317185975979446, -4.565824210632183, -40.83107569449773, -0.4668370252997335, -2.101413585919007, -2.68743347392435, -1.122523621554606, -1.47292264772989, -2.025711042439859, -18.11363957870291, -0.4182734279110673, -1.859568153332338, -2.369456538071264, -0.9969703627269644, -1.30339371341796, -1.786052825519624, -15.9634997130312, 2.471800883465118, -3.540458424932526, 3.422060580511445, -3.541254050038805, -3.540391022025961, 6.411246397998916, -9.195148950538366, 8.887682572148334, -9.197254623193142, -9.194973231029961, 6.344422918987287, -9.084523208241853, 8.780717096905159, -9.086556981860827, -9.084346462286842, 7.901994436038501, -11.31775215445371, 10.93925426923566, -11.32028861973358, -11.31753692381312, 6.824087153681296, -9.791945791265167, 9.464566952820924, -9.794210068802167, -9.791761809088843, 7.417828226618781, -10.62212564460754, 10.26691485810006, -10.62451446200199, -10.62192154636728},
        //        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        //        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
        //    };
        //    if (!approx_equal(tmp, expected_tmp, "reldiff", 0.000000000000001))
        //    {
        //        print_mat(dudq_dw, "dudq_dw");
        //        print_mat(dnet_dw2, "dnet_dw2");
        //        print_mat(dnet_didq2, "dnet_didq2");
        //        print_mat(didq_dw, "didq_dw");
        //        print_mat(dnet_dhist_err2, "dnet_dhist_err2");
        //        print_mat(didq_dw_matrix_sum, "didq_dw_matrix_sum");
        //    }
        //}    

        //if (id == 0)
        //{
        //    cout << trajectory_number << ", " << len << endl;
        //    print_mat(tmp, "tmp");    
        //}

        //didq_dw = a * didq_dw + b * tmp;
        //didq_dw_matrix_sum = didq_dw_matrix_sum + didq_dw.rows(0, 1);
        //}

        idq = a * idq + b * join_vert(vdq, udq, colvec({0, 0}));

        // add saturation to dq currents
        // idq(1:5) = max(min(idq(1:4), 1000*ones(4,1)),-1000*ones(4,1));
        idq.rows(0, 3) = arma::max(arma::min(idq.rows(0, 3), 1000*mat(4, 1, arma::fill::ones)), -1000 * mat(4, 1, arma::fill::ones));

        idq_his.col(len) = idq;
        idq_refi = calculate_idq_ref(trajectory_number, len+1, ts, preloaded_idq_refs);
        j_matrix.row(len+1) = (idq.rows(0, 1) - idq_refi).t() * didq_dw.rows(0, 1) * cost_term_power * pow(arma::sum((idq.rows(0, 1) - idq_refi) % (idq.rows(0, 1) - idq_refi)), cost_term_power / 2.0 - 1);

    }
    //}

    j_total = arma::sum(e_hist_err % e_hist_err);
    j_matrix.shed_row(j_matrix.n_rows-1);
}

// ============================================================================
int main ( int argc, char *argv[] )
{
    if ( argc != 3 )
    {
        cout<<"usage: "<< argv[0] <<" <workeri> <num_sample_time>\n";
        return -1;
    }

    const int workeri = atoi(argv[1]);
    const int num_sample_time = atoi(argv[2]);

	const bool use_shortcuts = false;
	const bool use_idq = false;
	const double vd = 690;
	const colvec vdq = colvec({vd, 0.0});

	// const double fs = 60.0;

	const double t_final = 1;
	const double ts = 0.001;
	const double vdc = 1200;
	colvec idq_ref_centre = colvec({0, 0});

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
	// const double min_grad = 1e-5;

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

    mat preloaded_idq_refs;
	load_matrix_from_csv(preloaded_idq_refs, "total_idq_refs.csv");

	const int maxworkers = 50;
	const int numruns = 10;
	const int num_iterations = 1024;
	const int num_samples = 10 * num_sample_time;

	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;

	string filename("openmp-results.csv");
	ofstream file_out;
	file_out.open(filename, std::ios_base::app);

	mat idq_start_positions;
	colvec idq, dw, dw_y, rr;
	mat w1_temp, w2_temp, w3_temp;
	mat j_matrix, j_matrix2;
	rowvec e_hist_err, e_hist_err2;
	mat jj, ii, h_matrix;
	double j_total, j_total_sum, j_total2, j_total_sum2;
	bool p1, p2;

    omp_set_num_threads(workeri);

	idq_start_positions.clear();
	for (int istart=0; istart < num_sample_time; istart++)
	{
	    idq_start_positions = join_horiz(idq_start_positions, idq_start_positions_temp);
	}

	begin = std::chrono::steady_clock::now();
	for (int runi=0; runi < numruns; runi++)
	{
	    mat w1 = {
	        { 0.081472368639318, 0.027849821886705, 0.095716694824295, 0.079220732955955, 0.067873515485777 },
	        { 0.090579193707562, 0.054688151920498, 0.048537564872284, 0.095949242639290, 0.075774013057833 },
	        { 0.012698681629351, 0.095750683543430, 0.080028046888880, 0.065574069915659, 0.074313246812492 },
	        { 0.091337585613902, 0.096488853519928, 0.014188633862722, 0.003571167857419, 0.039222701953417 },
	        { 0.063235924622541, 0.015761308167755, 0.042176128262628, 0.084912930586878, 0.065547789017756 },
	        { 0.009754040499941, 0.097059278176062, 0.091573552518907, 0.093399324775755, 0.017118668781156 }
	    };

	    mat w2 = {
	        { 0.012649986532930, 0.031747977514944, 0.055573794271939, 0.055778896675488, 0.025779225057201, 0.040218398522248, 0.087111112191539 },
	        { 0.013430330431357, 0.031642899914629, 0.018443366775765, 0.031342898993659, 0.039679931863314, 0.062067194719958, 0.035077674488589 },
	        { 0.009859409271100, 0.021756330942282, 0.021203084253232, 0.016620356290215, 0.007399476957694, 0.015436980547927, 0.068553570874754 },
	        { 0.014202724843193, 0.025104184601574, 0.007734680811268, 0.062249725927990, 0.068409606696201, 0.038134520444447, 0.029414863376785 },
	        { 0.016825129849153, 0.089292240528598, 0.091380041077957, 0.098793473495250, 0.040238833269616, 0.016113397184936, 0.053062930385689 },
	        { 0.019624892225696, 0.070322322455629, 0.070671521769693, 0.017043202305688, 0.098283520139395, 0.075811243132742, 0.083242338628518 }
	    };

	    mat w3 = {
	        { 0.002053577465818, 0.065369988900825, 0.016351236852753, 0.079465788538875, 0.044003559576025, 0.075194639386745, 0.006418708739190 },
	        { 0.092367561262041, 0.093261357204856, 0.092109725589220, 0.057739419670665, 0.025761373671244, 0.022866948210550, 0.076732951077657 }
	    };
	    double num_weights = w1.size() + w2.size() + w3.size();

	    mu = 1.0;
	    for (int iteration = 1; iteration < num_iterations + 1; iteration++)
	    {
	    	rowvec hist_err_total = rowvec(trajectory_length * num_samples, arma::fill::zeros);
	     	mat j_matrix_total = mat(trajectory_length * num_samples, num_weights, arma::fill::zeros);
	    	j_total_sum = 0;
	        mat idq_his, idq_ref_his;

            // We use default(none) to make sure we've accounted for whether variables should be shared or private
            #pragma omp parallel default(none) private(idq, j_total, e_hist_err, j_matrix, idq_his, idq_ref_his) shared(idq_start_positions, preloaded_idq_refs, w1, w2, w3, j_total_sum, j_matrix_total, hist_err_total, cout)
            {
                #pragma omp for
                for (int i = 0; i < num_samples; i++)
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
                        preloaded_idq_refs,
                        j_total, e_hist_err, j_matrix,
                        idq_his, idq_ref_his
                    );

                    if (i < 10)
                    {
                        // Adding together a sum is dangerous without a critical section
                        #pragma omp critical
                        {
                            int id = omp_get_thread_num();
                            //cout << "Index " << i << " in thread " << id << " assigning this:" << endl;
                            //print_mat(j_matrix, "j_matrix");
                            j_total_sum += j_total;
                            hist_err_total.cols(i * (trajectory_length - 1), (i+1) * (trajectory_length - 1) - 1) = e_hist_err.cols(1, e_hist_err.n_cols-1);
                            j_matrix_total.rows(i * (trajectory_length - 1), (i+1) * (trajectory_length - 1) - 1) = j_matrix.rows(1, j_matrix.n_rows-1);
                        }

                    }
                }
            }
	        mat L = mat(num_weights, num_weights, arma::fill::zeros);
	        mat R = mat(num_weights, num_weights, arma::fill::zeros);

	        while (mu < mu_max)
	        {
	    	    jj = j_matrix_total.t() * j_matrix_total;
	    	    ii = -1 * j_matrix_total.t() * arma::vectorise(hist_err_total);
	    	    h_matrix = jj + mu * arma::eye(num_weights, num_weights);

                cout << "J_total_sum: " << j_total_sum << endl;
                print_mat(j_matrix_total.rows(81, 85).cols(81, 85), "j_matrix_total");
                return 0;

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

                //print_mat(w3_temp.rows(0, 5).cols(0, 5), "w3_temp");
                //print_mat(w2_temp.rows(0, 5).cols(0, 5), "w3_temp");
                //print_mat(w1_temp.rows(0, 5).cols(0, 5), "w3_temp");
                //return 0;
	    	    j_total_sum2 = 0;

                #pragma omp parallel default(none) private(idq, j_total2, e_hist_err2, j_matrix2, idq_his, idq_ref_his) shared(j_total_sum2, w3_temp, w2_temp, w1_temp, idq_start_positions, preloaded_idq_refs)
                {
                    #pragma omp for
                    for (int i = 0; i < num_samples; i++)
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
                            preloaded_idq_refs,
                            j_total2, e_hist_err2, j_matrix2,
                            idq_his, idq_ref_his
                        );

                        // Adding together a sum is dangerous without a critical section
                        #pragma omp critical
                        {
                            if (i < 10)
                            {
                                //cout << "J_total2: " << j_total2 << endl;
                                j_total_sum2 += j_total2;
                            }
                        }
                    }
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
	    	        std::cout << setprecision(16)
	    	    	<< "iteration: " << iteration
	    	    	<< ", mu=" << mu
	    	    	<< ", J_total_sum2=" << j_total_sum2 / trajectory_length / 10 << endl;
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
	        }

	        if (mu == mu_max)
	        {
	    	    std::cout << "reach mu_max " << endl;
	    	    break;
	        }
	    }
	}
	end = std::chrono::steady_clock::now();
	file_out << workeri << ", " << num_samples << ", " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << endl;
	file_out.close();
	return 0;
}

