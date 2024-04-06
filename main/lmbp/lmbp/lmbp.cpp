
// Built-in libraries 
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
#include <random>
#include <algorithm>

// Third party libraries here only                                                               
#include <mpi.h>

#include <armadillo>
// #include <Eigen/Core>
// #include <Eigen/Cholesky>

// my code here
#include "RNN.h"
#include "matrix_utility.h"

using namespace std;
using namespace arma;

//// ==================================================================
//void drop_existing(mat& W, vector<pair<int, int>>& dropped)
//{
//    pair<int, int> to_drop;
//    for (int i = 0; i < dropped.size(); i++)
//    {
//        to_drop = dropped.at(i);
//        W(to_drop.first, to_drop.second) = 0.0;
//    }
//    return;
//}
//
//// ==================================================================
//void drop_new(mat& W, vector<pair<int, int>>& dropped, const double dropout_percent, const double threshold)
//{
//    // decide how much to dropout 
//    int existing_dropouts = dropped.size();
//    int number_to_dropout = (int)(W.n_rows * (W.n_cols - 1) * dropout_percent);
//    int new_dropouts = max((number_to_dropout - existing_dropouts), 0);
//
//    // Return early if there are no more new_dropouts
//    if (new_dropouts == 0) { return; }
//
//    // find candidates for dropout
//    vector<pair<int, int>> candidates;
//    for (int r = 0; r < W.n_rows; r++)
//    {
//        // ignore last column since we aren't applying dropout on biases
//        for (int c = 0; c < W.n_cols - 1; c++)
//        {
//            if (W(r, c) <= threshold && W(r, c) != 0.0)
//            {
//                candidates.push_back(make_pair(r, c));
//            }
//        }
//    }
//
//    // and return early if there are no new candidates
//    if (candidates.size() == 0) { return; }
//
//    // shuffle the candidates
//    std::random_device rd;
//    std::default_random_engine rng(rd());
//    shuffle(candidates.begin(), candidates.end(), rng);
//
//    // drop the weights from candidates until we've dropped all the new ones or until we run out of candidates to drop
//    pair<int, int> to_drop;
//    while (new_dropouts != 0 && candidates.size() != 0)
//    {
//        to_drop = candidates.back();
//        candidates.pop_back();
//
//        W(to_drop.first, to_drop.second) = 0.0;
//
//        dropped.push_back(to_drop);
//        new_dropouts--;
//    }
//
//    return;
//}
//
//// ==================================================================
//void apply_dropout(mat& w1, mat& w2, mat& w3,
//    vector<pair<int, int>>& w1_dropped,
//    vector<pair<int, int>>& w2_dropped,
//    vector<pair<int, int>>& w3_dropped,
//    const double dropout_percent, const double threshold)
//{
//
//
//
//    // Drop existing sets existing weights that have already been marked to be dropped to 0.0 
//    drop_existing(w1, w1_dropped);
//    drop_existing(w2, w2_dropped);
//    drop_existing(w3, w3_dropped);
//
//    // Drop new sets new weights to be dropped and adds those to the dropped vectors for future calls to drop_existing
//    drop_new(w1, w1_dropped, dropout_percent, threshold);
//    drop_new(w2, w2_dropped, dropout_percent, threshold);
//    drop_new(w3, w3_dropped, dropout_percent, threshold);
//
//    return;
//}

// ============================================================================
void collect_timings(int rank, RNN &rnn, int total_workers, int num_samples, int num_sample_time)
{

	const mat idq_start_positions_temp = {
		{ 0.475860587278343, -0.002854960144321, -0.777698538311603, 0.879677164233489, -1.314723503486884, 0.064516742311104, -0.037533188819172, -1.725427789528692, 0.093108760058804, -0.430206242426100 },
		{ 1.412232686444504, 0.919867079806395, 0.566696097539305, 2.038876251414042, -0.416411219699434, 0.600291949185784, -1.896304493622450, 0.288228089665011, -0.378157056589758, -1.627322736469780 },
		{ 0.022608484309598, 0.149808732632761, -1.382621159480352, 0.923932448688937, 1.224687824785337, -1.361514954864567, -2.127976768182674, -1.594183720266807, -1.482676111059003, 0.166347492460066 },
		{ -0.047869410220206, 1.404933445676977, 0.244474675589888, 0.266917446595828, -0.043584205546333, 0.347592631960065, -1.176923330714958, 0.110218849223381, -0.043818585358295, 0.376265910450719 },
		{ 1.701334654274959, 1.034121539569710, 0.808438803167691, 0.641661506421779, 0.582423277447969, -0.181843218459334, -0.990532220484176, 0.787066676357980, 0.960825211682115, -0.226950464706233 },
		{ -0.509711712767427, 0.291570288770806, 0.213041698417000, 0.425485355625296, -1.006500074619336, -0.939534765941492, -1.173032327267405, -0.002226786313836, 1.738244932613340, -1.148912289618790 }
	};

	mat idq_start_positions;
	for (int istart = 0; istart < num_sample_time; istart++)
	{
		idq_start_positions = join_horiz(idq_start_positions, idq_start_positions_temp);
	}

	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;
	if (rank == 0)
	{
		begin = std::chrono::steady_clock::now();
	}

	const int numruns = 10;
    int max_iterations = 1024;
    bool verbose = true;
	for (int runi = 0; runi < numruns; runi++)
	{
		rnn.train_best_weights(max_iterations, verbose);
	}

	if (rank == 0)
	{
		string filename("results.csv");
		ofstream file_out;
		end = std::chrono::steady_clock::now();
		file_out.open(filename, std::ios_base::app);
		file_out << total_workers << ", " << num_samples << ", " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << endl;
		file_out.close();
    }
    return;
}

// ============================================================================
void train_test_split(mat &testing_start, mat &training_start)
{
	// Our RNN class is currently coded to determine the output of each idq. Normally that would be here as well. So technically, these are all inputs
	const mat idq_start_positions = {
		{ -6.84301333971075e-01,3.72245101032735e-01,1.82704345806577e-00,-3.73274264593881e-04,3.04101909740982e-01,-9.412504882954e-01,1.61603023159936e-00,1.53185752933478e-00,7.16313846530595e-01,-2.01807882427201e-01,8.28965673215982e-01,7.42454981286267e-02,1.72060963811149e-00,-1.4262978217439e-00,-9.89193446357095e-01,1.24712525655525e-00,-1.27922797709869e-00,1.35130201611512e-01,-4.30743447588089e-01,-3.02304512488057e-01,-5.40527312148466e-01,-1.26693910560773e-00,1.17189945830112e-00,-7.29246749788583e-01,-1.80386755101131e-00,-1.30383488343996e-00,-4.69196024145268e-01,-5.15930282831294e-01,-1.21080024680692e-00,-1.43282934434169e-00,-1.9291832755753e-00,1.78357365583069e-00,-1.14441750364205e-00,-6.97695451001457e-01,1.68791154572183e-00,1.68235472585226e-00,1.9831217255736e-00,8.11242517670169e-01,7.38430545581331e-01,-1.33235435393374e-00
 		},
		{ 9.02147934488229e-01,4.90627535178487e-01,-1.61616971097407e-00,-5.59558457590091e-01,-1.18559533222421e-00,-9.11092127807228e-01,-2.5060937711233e-01,-1.97427607476905e-00,-1.52310990218816e-00,-1.92850862632347e-00,-7.53791679442666e-01,-1.4740338004201e-00,-2.79105553544678e-01,-1.53351508629083e-00,7.02954226610457e-01,-7.8574260066808e-02,-7.48216960344738e-01,1.4106050709486e-00,1.15177014730599e-00,-3.85406311997696e-01,1.73621049870266e-00,9.55294060486938e-01,-5.30949647584571e-01,1.29795597522221e-00,1.47641493665235e-01,-1.45431934955391e-00,-5.26194697845296e-01,1.65394143896853e-00,-1.17404562625925e-00,1.92985376884418e-00,8.77503563560074e-01,1.61183611061873e-00,4.97294158333498e-01,-4.22663688768245e-01,-1.77366064399473e-00,1.9760277580523e-00,8.27615740784582e-03,1.00512972979794e-00,-9.26524018315645e-01,-8.56387240415813e-01
 		},
		{ -1.46856876009581e-00,-1.89746893974416e-01,7.920708500803e-01,1.56097801984109e-00,1.66403849671452e-00,3.6371438909839e-01,-1.14045583192314e-00,1.95757242771517e-00,1.47692296662619e-00,-1.16966459506589e-00,5.64649010495303e-01,-1.99226017702953e-00,7.58613301745912e-01,1.28608197533521e-00,1.51628787661382e-01,-1.45419053934562e-00,-9.56925273742552e-01,4.1172323217035e-01,7.53856751708889e-01,1.58659774478371e-00,-1.9449173445051e-00,1.38165012882298e-00,9.5548820181568e-01,1.00442706707926e-00,-1.03876986052674e-01,2.23461453496316e-01,1.11223642369916e-00,9.49463357785675e-01,-4.98327159640938e-01,-7.28751214510956E-01,-7.61957918156038e-01,-1.92261321367048e-00,1.881277139233e-00,2.58267868220974e-01,1.80309265716433e-00,-3.15894670948504e-01,4.24164438214675e-01,-7.29210698560834e-02,-1.76714982010767e-00,-4.4250989416667e-01
 		},
		{ 3.40968031203543e-01,-3.14600382539129e-01,1.75561170051497e-00,-1.48842621439711e-00,1.62869194707966e-01,-1.33733029247884e-01,7.41496455395694e-01,-1.79507045977144e-00,3.5567230357181e-01,1.96516989075669e-00,1.78420479593887e-00,9.37749294261423e-01,-1.606456151797e-03,-6.08288062166409e-01,-1.00766052745448e-00,-5.84486833164185e-01,-1.2933884262837e-00,-1.18116810032968e-00,-4.76857448168176e-01,5.3689887235193e-01,-1.72594080878977e-00,-9.68945927052336e-01,1.2899446677309e-00,-9.25637420203116e-01,9.86317357063282e-01,-9.08860114683436e-01,1.20193043698753e-00,1.86102992368126e-00,-1.57363226842187e-00,1.18416055830737e-00,-1.8153035779065e-00,1.88823472892789e-00,8.90537617001011e-01,3.163037586582e-01,-1.8846501357631e-00,-1.47426605977504e-00,2.14018890298802e-01,-1.5441618226253e-00,1.54065168823373e-00,8.74041192325681e-01
 		},
		{ -7.98822373059539e-01,1.63376775389637e-00,1.51689763428599e-00,1.58416803802252e-00,1.68151819815546e-01,-1.4322633481175e-00,1.6006518529093e-00,-6.40473572064492e-01,1.54320340095107e-00,-7.44188089841052e-01,2.10617363343323e-01,2.18052170576649e-01,1.88857061700831e-00,1.24740422623177e-00,-8.97859941044752e-01,5.33566561934459e-01,-1.0223586747394e-00,-2.34741237416473e-01,-8.21903548019356e-01,-8.86563699481061e-01,8.49146684809956e-01,5.4452186728212e-02,7.08804025944922e-01,-1.60146037312106e-00,-1.17381141138907e-00,1.16793705049623e-00,5.2118049176713e-02,1.87525287190283e-00,-9.14364843788725e-01,1.34095253667182e-00,4.98936545925218e-01,-1.53971657229523e-00,6.22705840076408e-01,1.30425880011018e-00,-4.96084948404778e-01,-1.97308188541522e-00,1.77705977689262e-01,1.5638298682654e-00,-6.21292304091544e-01,-8.23417495138896e-01
 		},
		{ -9.57251396514485e-01,-3.22379262650712e-01,-1.52827167088209e-00,-8.15438507498087e-01,1.14789204660169e-00,7.36205975678375e-01,9.6470853353186e-01,1.90732486655872e-00,2.58304436345079e-01,2.67472183226445e-01,-9.02095124696141e-01,2.74396493158413e-05,-1.1251209527255e-00,1.26240364829838e-00,-9.88569231286051e-01,-1.37546210666876e-00,8.72082363565731e-01,-2.80016996205472e-01,-2.040037753908e-01,1.40152156505505e-00,-1.24194800977701e-00,-1.67527230548537e-00,1.94158226430011e-00,-5.22539247091398e-01,1.86920043594084e-00,-9.40501431023373e-01,-1.61999015500049e-00,-1.88203181861015e-00,1.39626859258241e-00,1.86658948706146e-00,-6.49697680904179e-01,1.94835340726948e-00,1.88400376915963e-00,-3.52148976110252e-01,-1.14734592358781e-00,-1.79795650910825e-00,5.05043877594956e-01,1.86929982513082e-00,-4.20579487562301e-01,8.10266804838709E-01 }
	};

	// Let's train on nine samples and then test on the last sample only
	training_start = idq_start_positions.cols(0, 38);
	testing_start = idq_start_positions.col(39);
}

// ============================================================================
void generate_convergence(int rank, const mat &testdata, RNN &rnn, string filename) 
{
    int trajectory_length = rnn.get_trajectory_length();
    double ts = rnn.get_ts();

    int max_iterations = 1024;
    bool verbose = true;

	// Get the best weights
	rnn.train_best_weights(max_iterations, verbose);
	mat w1, w2, w3;
	rnn.get_weights(w1, w2, w3);

    if (rank == 0)
    {
		double j_total;
		rowvec e_hist_err;
		mat j_matrix, idq_his, idq_ref_his;

		// then, run model on testing sample
		rnn.unroll_trajectory_full(testdata, 9, trajectory_length, w3, w2, w1, j_total, e_hist_err, j_matrix, idq_his, idq_ref_his);

		// the idq_his and idq_ref_his is what we need to compare to see how well the model does on the tenth sample
		mat convergence_matrix = mat(trajectory_length, 5);
		convergence_matrix.col(0) = arma::linspace<colvec>(0, ts * trajectory_length, trajectory_length); // x axis == elapsed time
		convergence_matrix.col(1) = idq_ref_his.row(0).t(); // reference output #1 (AC or DC)
		convergence_matrix.col(2) = idq_ref_his.row(1).t(); // reference output #2 (AC or DC)
		convergence_matrix.col(3) = idq_his.row(0).t();     // model output #1 (AC or DC)
		convergence_matrix.col(4) = idq_his.row(1).t();     // model output #2 (AC or DC)

		util::save_matrix_to_csv(convergence_matrix, filename);
    }
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

		// FIXME: I feel like this shit is whack. The RNN class should handle all parallelization under the hood.
		// It shouldn't depend on an outside program right? Idk. Maybe its the only way with OpenMPI
		mat w1, w2, w3;
		RNN rnn = RNN(rank, workerids, active_workers, num_samples - 1, trainingdata);

		// -------------------------------------------------------------------------------------------
		// USE CASES START HERE
		// -------------------------------------------------------------------------------------------

		// -------------------------------------------------------------------------------------------
		// This one will print the results to the screen and it will save the amount of time it took to run numrun times to results.csv
		// collect_timings(rank, rnn, total_workers, num_samples, num_sample_time);
		// -------------------------------------------------------------------------------------------

		// -------------------------------------------------------------------------------------------
		// Show convergence without dropout
		//string filename = "convergence_no_dropout.csv";
		//generate_convergence(rank, testdata, rnn, filename);
		//rnn.get_weights(w1, w2, w3);

		//util::save_matrix_to_csv(w1, "w1_best_no_dropout.csv");
		//util::save_matrix_to_csv(w2, "w2_best_no_dropout.csv");
		//util::save_matrix_to_csv(w3, "w3_best_no_dropout.csv");

		// -------------------------------------------------------------------------------------------
		// Show convergence with dropout

		// we will permit the cost to grow slighly higher than the starting cost, in this case by 15
		double dropout_max_extra_cost = 15;


		int trajectory_length = rnn.get_trajectory_length();
		double ts = rnn.get_ts();

		int max_iterations = 1024;
		bool verbose = true;

		double j_total;
		rowvec e_hist_err;
		mat j_matrix, idq_his, idq_ref_his;

		mat test_costs = mat(11, 3);
		mat convergence_matrix = mat(trajectory_length, 5);

		// Load our best weights without implementing dropout
		util::load_matrix_from_csv(w1, "w1_best_no_dropout.csv");
		util::load_matrix_from_csv(w2, "w2_best_no_dropout.csv");
		util::load_matrix_from_csv(w3, "w3_best_no_dropout.csv");

		rnn.set_weights(w1, w2, w3);

		// So we can record the convergence / test cost without dropout
		rnn.unroll_trajectory_full(testdata, 9, trajectory_length, w3, w2, w1, j_total, e_hist_err, j_matrix, idq_his, idq_ref_his);
		test_costs.row(0) = rowvec({ 0.0, j_total, 0.0 });

		// Now we test with dropout here only
		rnn.set_dropout(true);

		string filename = "";
		int max_weights_to_drop = 10;

		for (int max_drop = 1; max_drop <= max_weights_to_drop; max_drop++)
		{
			cout << "Training again after dropping " + to_string(max_drop) + " weights..." << endl;

			filename = "convergence_" + to_string(max_drop) + "_dropped.csv";

			rnn.set_max_drop(max_drop);
			rnn.train_best_weights(max_iterations, verbose);
			rnn.get_weights(w1, w2, w3);

			if (rank == 0)
			{
				// then, run model on testing sample
				rnn.unroll_trajectory_full(testdata, 9, trajectory_length, w3, w2, w1, j_total, e_hist_err, j_matrix, idq_his, idq_ref_his);

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
				test_costs.row(max_drop) = rowvec({(double) max_drop, j_total, rnn.get_last_weight_dropped() });
			}
		}
		util::save_matrix_to_csv(test_costs, "test_costs.csv");
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

