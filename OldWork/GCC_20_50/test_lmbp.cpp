
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

#include <armadillo>

#include "functions.h"
using namespace std;
using namespace arma;

// ============================================================================
// a test script to test train_weights_using_lm
int main()
{
    const bool use_shortcuts = false;
    const bool use_idq = false;
    const double vd = 20;
    const double l = 0.025 * 1;
    const double r = 0.5 / 2;
    colvec vdq = colvec({vd, 0.0});

    const double fs = 60.0;
    const double ws = 2.0 * M_PI * fs;
    const double xl = ws * l;

    const double t_final = 1;
    const double ts = 0.001;
    const double vdc = 50.0;
    colvec idq_ref = colvec({0, 0});

    const double vmax = vdc * sqrt(3.0 / 2.0) / 2.0;
    const double imax = 3.0;
    const double iq_max = (vmax - vd) / xl;

    const double gain1 = 1000.0;
    const double gain2 = 0.5;
    const double gain3 = 0.5;

    const double cost_term_power = 1.0 / 2.0;

    const double mu = 0.001;
    const double mu_dec = 0.1;
    const double mu_inc = 10;
    const double mu_max = 1e10;
    const double mu_min = 1e-20;
    const double min_grad = 1e-6;

    const double num_iterations = pow(2, 7);
    const double trajectory_length = t_final / ts;
    const double num_samples = 1;

    string basedir = "./testfiles/test_lmbp_changes/";
    mat a, b, c, d;
    load_matrix_from_csv(a, basedir + "a.csv");
    load_matrix_from_csv(b, basedir + "b.csv");
    load_matrix_from_csv(c, basedir + "c.csv");
    load_matrix_from_csv(d, basedir + "d.csv");

    std::ofstream output(basedir + "cpp_output.txt");
    std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
    // std::streambuf *cerrbuf = std::cerr.rdbuf(); //save old buf
    std::cout.rdbuf(output.rdbuf());
    // std::cerr.rdbuf(output.rdbuf());

    int numruns = 100;
    for (int i = 1; i < numruns+1; i++)
    {
        cout << "Run: " << i << endl;
        string currentdir = basedir + "runs/" + to_string(i);

        // we fix these values because they are generated randomly otherwise
        mat idq_start_positions, idq_ref_total;
        load_matrix_from_csv(idq_start_positions, currentdir + "/idq_startpositions.csv");
        load_matrix_from_csv(idq_ref_total, currentdir + "/idq_ref_total.csv");

        mat w1, w2, w3;
        mat ending_w1, ending_w2, ending_w3;

        // load the initial weights for this test case
        load_matrix_from_csv(w1, currentdir + "/starting_w1.csv");
        load_matrix_from_csv(w2, currentdir + "/starting_w2.csv");
        load_matrix_from_csv(w3, currentdir + "/starting_w3.csv");

        train_weights_using_lm(
            w1, w2, w3,
            mu, use_shortcuts, use_idq,
            num_samples, trajectory_length, num_iterations,
            vdq, mu_max, mu_inc, mu_dec, mu_min, min_grad,
            ts, imax, iq_max, vmax, xl,
            gain1, gain2, gain3,
            cost_term_power,
            a, b, c, d, idq_start_positions, idq_ref_total
        );
    }

    // std::cerr.rdbuf(cerrbuf); //reset to standard error again
    std::cout.rdbuf(coutbuf); //reset to standard output again
    return 0;
}
