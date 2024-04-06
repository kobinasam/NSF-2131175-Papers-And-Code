
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <armadillo>

#include "functions.h"

using namespace std;
using namespace arma;

int main()
{
    // FIXME: Would be better to generate permutations rather than writing this out manually
    vector<vector<bool>> condition_bools = {
        {true, true, true},
        {true, true, false},
        {true, false, true},
        {true, false, false},
        {false, true, true},
        {false, true, false},
        {false, false, true},
        {false, false, false}
    };

    for (int i = 0; i < condition_bools.size(); i++)
    {
        // globals needed to call unrollTrajectoryFull
        double vdc=50;
        double vmax = vdc*sqrt(3.0/2.0)/2.0;
        double vd=20;
        colvec vdq = colvec({vd, 0});

        double gain1=1000;
        double gain2=100*2;
        double gain3=100*2;

        mat a = mat({
            {0.920525055277549, 0.364461652184452},
            {-0.364461652184452, 0.920525055277549}
        });

        mat b = mat({
            {-0.038866915258523, -0.007401576588668},
            {0.007401576588668, -0.038866915258523}
        });

        double t_final = 1;
        double ts = 0.001;
        double cost_term_power = 1.0/2.0;

        // arguments needed to call unrollTrajectoryFull
        double num_samples = 1;
        colvec idq = colvec({1.0753, 3.6678});
        double trajectory_number = 1;
        double trajectory_length = t_final / ts;

        // variable inputs of unrollTrajectoryFull
        bool use_shortcuts = condition_bools[i][0];
        bool use_idq = condition_bools[i][1];
        bool flag = condition_bools[i][2];

        vector<pair<string, bool>> conditions = {
            {"shortcuts", use_shortcuts},
            {"use_idq", use_idq},
            {"flag", flag}
        };

        string basedir = "./testfiles/test_unrolltrajectory";
        string subdir = generate_subdir(conditions);
        string filepath = basedir + "/" + subdir + "/";

        mat w1, w2, w3;
        load_matrix_from_csv(w1, filepath + "w1.csv");
        load_matrix_from_csv(w2, filepath + "w2.csv");
        load_matrix_from_csv(w3, filepath + "w3.csv");

        // more globals
        mat idq_ref_total=mat(2 * num_samples, trajectory_length + 1, arma::fill::zeros);
        double num_weights = w1.n_elem + w2.n_elem + w3.n_elem;

        // outputs of unroll_trajectory_full
        mat e_hist_err, j_matrix;
        double j_total;

        // Expected outputs
        mat expected_j_matrix, expected_j_total, expected_e_hist_err;
        load_matrix_from_csv(expected_j_total, filepath + "j_total.csv");
        load_matrix_from_csv(expected_e_hist_err, filepath + "e_hist_err.csv");

        unroll_trajectory_full(
            idq, trajectory_number, trajectory_length, w3, w2, w1, flag, use_shortcuts, use_idq,
            idq_ref_total, a, b, vmax, ts, gain1, gain2, gain3, cost_term_power,
            j_total, e_hist_err, j_matrix);

        cout << "Testing " << filepath << endl;
        mat j_total_matrix = mat(1, 1);
        j_total_matrix(0, 0) = j_total;

        show_digits_of_diff(j_total_matrix, expected_j_total, "J_TOTAL");
        show_digits_of_diff(e_hist_err, expected_e_hist_err, "E HIST ERR");
        if (flag)
        {
            load_matrix_from_csv(expected_j_matrix, filepath + "j_matrix.csv");
            show_digits_of_diff(j_matrix, expected_j_matrix, "J_MATRIX");
        }
        cout << endl;
    }
}