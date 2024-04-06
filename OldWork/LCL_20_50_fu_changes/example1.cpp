
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
	colvec vdq = colvec({ vd, 0.0 });

	const double fs = 60.0;

	const double t_final = 1;
	const double ts = 0.001;
	const double vdc = 50.0;
	colvec idq_ref = colvec({ 0, 0 });
	colvec idq_ref_centre = colvec({ 0, 0 });

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

	const double num_iterations = 200;
	const double trajectory_length = t_final / ts;
	const int num_samples = 10;

	const int num_hids1 = 6;
	const int num_hids2 = 6;
	const int num_outputs = 2;
	const int num_inputs = use_idq ? 10 : 4;

	// FIXME: For sanity, don't run this many times...
	// int numruns = 100000000;
	int numruns = 20;
	mat w1, w2, w3;
	for (int i = 1; i < numruns + 1; i++)
	{
        // Action network initialization: 2*6*6*2 with tanh functions for hidden and output layers
        if (use_shortcuts)
        {
            w1 = 0.1 * mat(num_hids1, num_inputs + 1, arma::fill::randn);
            w2 = 0.1 * mat(num_hids2, num_hids1 + num_inputs + 1, arma::fill::randn);
            w3 = 0.1 * mat(num_outputs, num_hids2 + num_hids1 + num_inputs + 1, arma::fill::randn);
        }
        else
        {
            arma_rng::set_seed_random();
            w1 = 0.1 * mat(num_hids1, num_inputs + 1, arma::fill::randn);
            w2 = 0.1 * mat(num_hids2, num_hids1 + 1, arma::fill::randn);
            w3 = 0.1 * mat(num_outputs, num_hids2 + 1, arma::fill::randn);
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
	system("pause");
	return 0;
}
