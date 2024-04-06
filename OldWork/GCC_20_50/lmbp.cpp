
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <armadillo>

#include "functions.h"

using namespace std;
using namespace arma;

// ============================================================================
// A rewrite of LMBP.m
int main()
{
    // Data initialization
    bool use_shortcuts = false;
    bool use_idq = false;


    double vd = 20;
    double l = 0.025 * 1;
    double r = 0.5 / 2;
    colvec vdq = colvec({vd, 0.0});

    double fs = 60.0;
    double ws = 2.0 * M_PI * fs;
    double xl = ws * l;

    double t_final = 1;
    double ts = 0.001;
    double vdc = 50.0;
    colvec idq_ref = colvec({0, 0});

    double vmax = vdc * sqrt(3.0 / 2.0) / 2.0;
    double imax = 3.0;
    double iq_max = (vmax - vd) / xl;

    double gain1 = 1000.0;
    double gain2 = 100 * 2.0;
    double gain3 = 100 * 2.0;

    double cost_term_power = 1.0 / 2.0;

    double mu = 0.001;
    double mu_dec = 0.1;
    double mu_inc = 10;
    double mu_max = 1e10;
    double mu_min = 1e-20;
    double min_grad = 1e-6;

    double num_iterations = pow(2, 7);
    double trajectory_length = t_final / ts;
    double num_samples = 1;

    // FIXME: For now, we do not implement the logic to compute these, so load from matlab
    mat a, b, c, d;
    load_matrix_from_csv(a, "./testfiles/a.csv");
    load_matrix_from_csv(b, "./testfiles/b.csv");
    load_matrix_from_csv(c, "./testfiles/c.csv");
    load_matrix_from_csv(d, "./testfiles/d.csv");

    // Action network initialization: 2*6*6*2 with tanh functions for hidden and output layers
    double num_hids1 = 6;
    double num_hids2 = 6;
    double num_outputs = 2;
    double num_inputs = use_idq ? 6: 4;

    mat w1, w2, w3, w1_temp, w2_temp, w3_temp;
    if (use_shortcuts)
    {
        w1 = mat(num_hids1, num_inputs + 1);
        w2 = mat(num_hids2, num_hids1 + num_inputs + 1);
        w3 = mat(num_outputs, num_hids2 + num_hids1 + num_inputs + 1);
    }
    else
    {
        w1 = mat(num_hids1, num_inputs + 1);
        w2 = mat(num_hids2, num_hids1 + 1);
        w3 = mat(num_outputs, num_hids2 + 1);
    }

    double num_weights = w1.size() + w2.size() + w3.size();

    double rss, rss_with_mu;

    colvec idq, dw, dw_y, rr;
    mat e_hist_err, j_matrix, jj, ii, h_matrix, L, R;
    double j_total, j_total_sum, j_total2, j_total_sum2;
    bool p1, p2;

    mat hist_err_total = mat(1, trajectory_length * num_samples, arma::fill::zeros);
    mat j_matrix_total = mat(trajectory_length * num_samples, num_weights, arma::fill::zeros);

    for (int j = 0; j < pow(2, 30); j++)
    {
        mu = 0.001;
        rss = 0;
        rss_with_mu = 0;

        w1 = 0.1 * w1.transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );
        w2 = 0.1 * w2.transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );
        w3 = 0.1 * w3.transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );

        train_weights_using_lm(
            w1, w2, w3,
            mu, use_shortcuts, use_idq,
            num_samples, trajectory_length, num_iterations,
            vdq,
            mu_max, mu_inc, mu_dec, mu_min, min_grad,
            ts, imax, iq_max, vmax, xl,
            gain1, gain2, gain3, cost_term_power,
            a, b, c, d
        );

        // if (j_total_sum2 / trajectory_length / num_samples < 0.1)
        // {
            // save(strcat('w',num2str(iteration),'Lm_',strrep(num2str((J_total_sum2/trajectoryLength/numSamples),3),'.','_')), 'W1', 'W2', 'W3');
            // save(strcat('R',num2str(iteration),'Lm_',strrep(num2str((J_total_sum2/trajectoryLength/numSamples),3),'.','_')), 'RR');
        // }
    }

// %  loglog(RR);grid;
// % save w5320Lm.mat W1 W2 W3;
}

