
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

#include <armadillo>
#include <omp.h>

#include "functions.h"
using namespace std;
using namespace arma;

// ============================================================================
bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

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

    const double t_final = 1;
    const double ts = 0.001;
    const double vdc = 50.0;
    colvec idq_ref = colvec({0, 0});
    colvec idq_ref_centre = colvec({0, 0});

    const double vmax = vdc * sqrt(3.0 / 2.0) / 2.0;

    const double gain1 = 1000.0;
    const double gain2 = 0.5;
    const double gain3 = 0.5;

    const double cost_term_power = 1.0 / 2.0;

    const double mu = 0.001;
    const double mu_dec = 0.1;
    const double mu_inc = 10;
    const double mu_max = 1e10;
    const double mu_min = 1e-20;

    const double trajectory_length = t_final / ts;
    const int num_samples = 10;

    const int num_hids1 = 6;
    const int num_hids2 = 6;
    const int num_outputs = 2;
    const int num_inputs = use_idq ? 10: 4;

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    string filename("results.csv");
    ofstream file_out;
    file_out.open(filename, std::ios_base::app);
    file_out << "Num Threads" << ", " << "Microseconds" << endl;

    // Use as many cores as possible, but don't exceed num_samples (that's what we are parallelizing over)
    int maxcores = min(omp_get_num_procs(), num_samples);

    mat w1, w2, w3;

    // Adjust numruns and num_iterations as needed
    int numruns = 1;
    const double num_iterations = 10;

    for (int numthreads=1; numthreads <= maxcores; numthreads++)
    {
        omp_set_num_threads(numthreads);
        begin = std::chrono::steady_clock::now();

        for (int i = 0; i < numruns; i++)
        {
            string currentdir = TESTDIR + "run" + std::to_string(i + 1) + "/";
            mkdir_from_string(currentdir);

            // Action network initialization: 2*6*6*2 with tanh functions for hidden and output layers
            if (TESTMODE)
            {
                load_matrix_from_csv(w1, currentdir + "w1.csv");
                load_matrix_from_csv(w2, currentdir + "w2.csv");
                load_matrix_from_csv(w3, currentdir + "w3.csv");
            }
            else
            {
                arma_rng::set_seed_random();
                if (use_shortcuts)
                {
                    w1 = 0.1 * mat(num_hids1, num_inputs + 1, arma::fill::randn);
                    w2 = 0.1 * mat(num_hids2, num_hids1 + num_inputs + 1, arma::fill::randn);
                    w3 = 0.1 * mat(num_outputs, num_hids2 + num_hids1 + num_inputs + 1, arma::fill::randn);
                }
                else
                {
                    w1 = 0.1 * mat(num_hids1, num_inputs + 1, arma::fill::randn);
                    w2 = 0.1 * mat(num_hids2, num_hids1 + 1, arma::fill::randn);
                    w3 = 0.1 * mat(num_outputs, num_hids2 + 1, arma::fill::randn);
                }
                // Only do this once if we haven't saved matrices for our runs
                // save_matrix_to_csv(w1, currentdir + "w1.csv");
                // save_matrix_to_csv(w2, currentdir + "w2.csv");
                // save_matrix_to_csv(w3, currentdir + "w3.csv");
            }

            train_weights_using_lm(
                w1, w2, w3,
                mu, use_shortcuts, use_idq,
                num_samples, trajectory_length, num_iterations,
                vdq, mu_max, mu_inc, mu_dec, mu_min,
                ts, vmax,
                gain1, gain2, gain3,
                cost_term_power
            );
        }
        std::cout << "Time executing " << numthreads << " threads:" << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "(seconds)" << std::endl;
        end = std::chrono::steady_clock::now();
        file_out << numthreads << ", " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << endl;
    }
    file_out.close();
    return 0;
}
