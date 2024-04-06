// Built-in libraries here
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <math.h>
#include <map>
#include <iostream>
#include <sstream>
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
#include <random>
#include <algorithm>
#include <chrono>

// Third party libraries
#include <mpi.h>

#include <armadillo>
// my code here
#include "RNN.h"
#include "matrix_utility.h"

using namespace std;
using namespace arma;

// ============================================================================
// void collect_timings(int rank, RNN &rnn, int total_workers, int num_samples, int num_sample_time)
// {
// 	// Here we initialize the random input buy generating random numbers
// 	class RandomNumberGenerator {
// 		public:
// 			static arma::mat generateRandomNumbers() {
// 				arma::arma_rng::set_seed(std::time(nullptr));
// 				return arma::randu<arma::mat>(6, 60);
// 			}
// 	};
// 	arma::mat idq_start_positions_temp = RandomNumberGenerator::generateRandomNumbers();
// 	//util::save_matrix_to_csv(idq_start_positions_temp, "myData1.csv");
// 	// const mat idq_start_positions_temp = {
// 	// 	{ 0.475860587278343, -0.002854960144321, -0.777698538311603, 0.879677164233489, -1.314723503486884, 0.064516742311104, -0.037533188819172, -1.725427789528692, 0.093108760058804, -0.430206242426100 },
// 	// 	{ 1.412232686444504, 0.919867079806395, 0.566696097539305, 2.038876251414042, -0.416411219699434, 0.600291949185784, -1.896304493622450, 0.288228089665011, -0.378157056589758, -1.627322736469780 },
// 	// 	{ 0.022608484309598, 0.149808732632761, -1.382621159480352, 0.923932448688937, 1.224687824785337, -1.361514954864567, -2.127976768182674, -1.594183720266807, -1.482676111059003, 0.166347492460066 },
// 	// 	{ -0.047869410220206, 1.404933445676977, 0.244474675589888, 0.266917446595828, -0.043584205546333, 0.347592631960065, -1.176923330714958, 0.110218849223381, -0.043818585358295, 0.376265910450719 },
// 	// 	{ 1.701334654274959, 1.034121539569710, 0.808438803167691, 0.641661506421779, 0.582423277447969, -0.181843218459334, -0.990532220484176, 0.787066676357980, 0.960825211682115, -0.226950464706233 },
// 	// 	{ -0.509711712767427, 0.291570288770806, 0.213041698417000, 0.425485355625296, -1.006500074619336, -0.939534765941492, -1.173032327267405, -0.002226786313836, 1.738244932613340, -1.148912289618790 }
// 	// };

// 	mat idq_start_positions;
// 	for (int istart = 0; istart < num_sample_time; istart++)
// 	{
// 		idq_start_positions = join_horiz(idq_start_positions, idq_start_positions_temp);
// 	}

// 	std::chrono::steady_clock::time_point begin;
// 	std::chrono::steady_clock::time_point end;
// 	if (rank == 0)
// 	{
// 		begin = std::chrono::steady_clock::now();
// 	}

// 	const int numruns = 10;
//     int max_iterations = 2192;
//     bool verbose = true;
// 	for (int runi = 0; runi < numruns; runi++)
// 	{
// 		rnn.train_best_weights(max_iterations, verbose);
// 	}

// 	if (rank == 0)
// 	{
// 		string filename("results.csv");
// 		ofstream file_out;
// 		end = std::chrono::steady_clock::now();
// 		file_out.open(filename, std::ios_base::app);
// 		file_out << total_workers << ", " << num_samples << ", " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << endl;
// 		file_out.close();
//     }
//     return;
// }


// ============================================================================
void train_test_split(mat &testing_start, mat &training_start)
{
	// Our RNN class is currently coded to determine the output of each idq. Normally that would be here as well. So technically, these are all inputs

	class RandomNumberGenerator {
		public:
			static arma::mat generateRandomNumbers() {
				arma::arma_rng::set_seed(std::time(nullptr));
				arma::mat randomNumbers = arma::randu<arma::mat>(6, 10);
				return randomNumbers; 
    		}
	};
	 //arma::mat idq_start_positions = RandomNumberGenerator::generateRandomNumbers();
	 //util::save_matrix_to_csv(idq_start_positions, "myData.csv");
	// exit(0);
	
	
	const mat idq_start_positions = {
		{ 5.0953506848797347e-01,2.5033276978957908e-01,1.4694861708564197e-01,3.9641116675598498e-01,8.1225006837552316e-01,3.2794145078767933e-01,3.1974827580379545e-01,7.4235391050275577e-03,6.2655906345827217e-01,5.0453160927408780e-01},
		{ 9.4980313617228740e-01,8.3362125893016159e-01,3.4987756430126510e-01,8.0695544556958043e-01,2.4925324721699976e-01,7.9705588264281604e-02,6.1486932492974222e-01,2.4565973308675834e-01,7.9533943431543441e-01,1.7132869191444958e-01},
		{ 7.1911031921027879e-02,5.4834035754923749e-01,7.1816549330833956e-01,8.4205284373763056e-01,3.0214103425534283e-01,9.2385066966052620e-01,7.1744265019091502e-01,1.2957413915475358e-01,2.2258053977700135e-01,3.9763349715510515e-01 },
		{ 6.5538435435336784e-01,5.1862559924015084e-01,7.0276191506160579e-01,6.9320234246203161e-01,8.4618656343432719e-01,6.7435017198383240e-01,5.2820619150061687e-01,4.8369343485070659e-01,7.5009973478789627e-01,8.0565607222518709e-01},
		{ 5.2172700541280530e-02,6.3016388593019579e-01,8.3687919445565273e-01,3.1103420458860342e-01,8.7662807278664490e-01,7.8332761214368363e-01,5.4744472906297825e-01,8.5379043506802288e-01,1.4048521418920509e-01,4.8852064804371625e-01},
		{ 9.6594100977429076e-01,2.4966696292260437e-01,6.5590081468131445e-01,8.3371068788011357e-02,4.4045695943933366e-01,6.6162026700074428e-01,5.3902386060849916e-01,3.1221302936765533e-01,3.3112411973020328e-01,6.4281958923029392e-01}
	};


	// Let's train on last samples and then test on the last sample
	training_start = idq_start_positions.cols(0,8);
	testing_start = idq_start_positions.col(9);
	// util::save_matrix_to_csv(training_start, "myData1.csv");
	// util::save_matrix_to_csv(testing_start, "myData2.csv");
	// exit(0);
}


// ============================================================================
int main(int argc, char* argv[])
{
	auto start = std::chrono::high_resolution_clock::now();

    if (argc != 2)
    {
        cout << "usage: " << argv[0] << "<num_sample_time>\n";
        return -1;
    }

    const int num_sample_time = atoi(argv[1]);

    int rank;
    int mpi_err;
    int total_workers;

    mpi_err = MPI_Init(NULL, NULL);
    if (mpi_err != 0)
    {
        cout << endl;
        cout << "HELLO_MPI - Fatal error!" << endl;
        cout << "MPI_Init returned nonzero error." << endl;
        exit(1);
    }

    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &total_workers);
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int num_samples = 10 * num_sample_time;

    // We have no need for more workers than we have samples, so use active_workers
    const int active_workers = min(num_samples, total_workers);

    vector<int> workerids;
    for (int i = 1; i < active_workers; i++)
    {
        workerids.push_back(i);
    }

    if (rank == 0)
    {
        cout << "Total workers: " << total_workers << endl;
        cout << "Active workers: " << active_workers << endl;
        cout << "Num Samples: " << num_samples << endl;
    }

	try
	{

		mat testdata, trainingdata;
		train_test_split(testdata, trainingdata);
		// cout << testdata<< endl;
		// cout << "------------------------------------------------------------"<< endl;
		// cout << trainingdata<< endl;
		// exit(0);

		// FIXME: I feel like this shit is whack. The RNN class should handle all parallelization under the hood.
		// It shouldn't depend on an outside program right? Idk. Maybe its the only way with OpenMPI
		int trajectory_length = 1000;
		double ts = 0.00;
		int max_iterations = 1024;

		double j_total;
		rowvec e_hist_err;
		mat w1, w2, w3;
		mat j_matrix, idq_his, idq_ref_his;

		mat convergence_matrix = mat(trajectory_length, 5);

		// -------------------------------------------------------------------------------------------
		// USE CASES START HERE
		// -------------------------------------------------------------------------------------------

		// -------------------------------------------------------------------------------------------
		// This one will print the results to the screen and it will save the amount of time it took to run numrun times to results.csv
		//collect_timings(rank, rnn, total_workers, num_samples, num_sample_time);
		// -------------------------------------------------------------------------------------------

		// -------------------------------------------------------------------------------------------
		// Show convergence without dropout
		if (true)
		{
			RNN rnn = RNN(rank, workerids, active_workers, num_samples - 1, trainingdata);
			// Train best weights
			rnn.train_best_weights(max_iterations, true);
			rnn.get_weights(w1, w2, w3);

			if (rank == 0)
			{
				// then, run model on testing sample
				rnn.unroll_trajectory_full(testdata, 5, trajectory_length, w3, w2, w1, j_total, e_hist_err, j_matrix, idq_his, idq_ref_his);

				// the idq_his and idq_ref_his is what we need to compare to see how well the model does on the tenth sample
				convergence_matrix.col(0) = arma::linspace<colvec>(0, ts * trajectory_length, trajectory_length); // x axis == elapsed time
				convergence_matrix.col(1) = idq_ref_his.row(0).t(); // reference output #1 (AC or DC)
				convergence_matrix.col(2) = idq_ref_his.row(1).t(); // reference output #2 (AC or DC)
				convergence_matrix.col(3) = idq_his.row(0).t();     // model output #1 (AC or DC)
				convergence_matrix.col(4) = idq_his.row(1).t();     // model output #2 (AC or DC)

				util::save_matrix_to_csv(convergence_matrix, "convergence_no_dropout.csv");
				
			}
			rnn.get_weights(w1, w2, w3);

			util::save_matrix_to_csv(w1, "w1_best_no_dropout.csv");
			util::save_matrix_to_csv(w2, "w2_best_no_dropout.csv");
			util::save_matrix_to_csv(w3, "w3_best_no_dropout.csv");
		}

		// -------------------------------------------------------------------------------------------
		// Show convergence with dropout

		// Load our best weights without implementing dropout
		if (true)
		{
			util::load_matrix_from_csv(w1, "w1_best_no_dropout.csv");
			util::load_matrix_from_csv(w2, "w2_best_no_dropout.csv");
			util::load_matrix_from_csv(w3, "w3_best_no_dropout.csv");

			RNN rnn = RNN(rank, workerids, active_workers, num_samples - 1, trainingdata);
			rnn.set_weights(w1, w2, w3);

			// So we can record the convergence / test cost without dropout
			rnn.unroll_trajectory_full(testdata, 5, trajectory_length, w3, w2, w1, j_total, e_hist_err, j_matrix, idq_his, idq_ref_his);

			string filename = "";
			int max_weights_to_drop = 20;

			mat test_costs = mat(max_weights_to_drop + 1, 3);
			test_costs.row(0) = rowvec({ 0.0, j_total / trajectory_length / 10, 0.0 });

			// Now we test with dropout
			rnn.set_dropout(true);


			for (int max_drop = 1; max_drop <= max_weights_to_drop; max_drop++)
			{
				cout << "Training again after dropping " + to_string(max_drop) + " weights..." << endl;

				filename = "convergence_" + to_string(max_drop) + "_dropped.csv";

				rnn.set_max_drop(max_drop);
				rnn.train_best_weights(max_iterations, true);
				rnn.get_weights(w1, w2, w3);

				if (rank == 0)
				{
					// then, run model on testing sample
					rnn.unroll_trajectory_full(testdata, 5, trajectory_length, w3, w2, w1, j_total, e_hist_err, j_matrix, idq_his, idq_ref_his);

					// the idq_his and idq_ref_his is what we need to compare to see how well the model does on the tenth sample
					convergence_matrix.col(0) = arma::linspace<colvec>(0, ts * trajectory_length, trajectory_length); // x axis == elapsed time
					convergence_matrix.col(1) = idq_ref_his.row(0).t(); // reference output #1 (AC or DC)
					convergence_matrix.col(2) = idq_ref_his.row(1).t(); // reference output #2 (AC or DC)
					convergence_matrix.col(3) = idq_his.row(0).t();     // model output #1 (AC or DC)
					convergence_matrix.col(4) = idq_his.row(1).t();     // model output #2 (AC or DC)

					util::save_matrix_to_csv(convergence_matrix, filename);
					util::save_matrix_to_csv(w1, "w1_best_" + to_string(max_drop) + "_dropped.csv");
					util::save_matrix_to_csv(w2, "w2_best_" + to_string(max_drop) + "_dropped.csv");
					util::save_matrix_to_csv(w3, "w3_best_" + to_string(max_drop) + "_dropped.csv");
					test_costs.row(max_drop) = rowvec({ (double)max_drop, j_total / trajectory_length / 10, rnn.get_last_weight_dropped() });
				}
			}
			util::save_matrix_to_csv(test_costs, "test_costs.csv");

		}
		
	}
	
	catch (const std::exception &e)
	{
		cout << "Errored with " << e.what() << endl;
	}
	MPI_Finalize();
	auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration = end - start;
			std::ofstream outFile("ExecutionTime.txt");
			if (outFile.is_open()) {
				outFile << "Execution Time: " << duration.count() << " seconds";
				outFile.close();
				std::cout << "Duration saved to ExecutionTime.txt\n";
			} else {
				std::cout << "Unable to open the file for writing.\n";
			}
			std::cout << "Execution Time: " << duration.count() << " seconds\n";
    return 0;
}

