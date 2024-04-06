
// built-ins
#include<vector>

// third-party libraries here
#include <mpi.h>
#include <armadillo>
#include "../eigen/Eigen/Core"
#include "../eigen/Eigen/Cholesky"

// my modules here
#include "RNN.h"
#include "matrix_utility.h"

// Definitions
#define _USE_MATH_DEFINES

#undef ARMA_OPENMP_THREADS
#define ARMA_OPENMP_THREADS 50 // Max threads used by armadillo for whatever logic they have parallelized. Cool
#define ARMA_NO_DEBUG          // Only define this if we want to disable debugging for production code

// Used to distingish message types in sending/receiving messages
#define MPI_J_TOTAL 0
#define MPI_NEW_JOB 1
#define MPI_JOB_FINISHED 2
#define MPI_LOOP_DONE 3

using namespace std;
using namespace arma;

RNN::RNN(
	int rank, 
	vector<int> workerids, 
	int active_workers, 
	int num_samples,
	const mat idq_start_positions,
	const bool add_dropout,
	const double dropout_max_extra_cost,
	const int max_drop,
	const bool use_shortcuts, 
	const bool use_idq, 
	const double vd, 
	const double t_final, 
	const double ts, 
	const double vdc, 
	const colvec idq_ref_centre,
	const double gain1,
	const double gain2,
	const double gain3,
	const double cost_term_power,
	const double starting_mu,
	const double mu_dec,
	const double mu_inc,
	const double mu_max,
	const double mu_min,
	const mat a,
	const mat b,
	const mat starting_w1,
	const mat starting_w2,
	const mat starting_w3)
{
	_rank = rank;
	_workerids = workerids;
	_active_workers = active_workers;
	_num_samples = num_samples;
    _use_shortcuts = use_shortcuts;
    _use_idq = use_idq;
    _vd = vd;
    _t_final = t_final;
    _ts = ts;
    _trajectory_length = t_final / ts;
    _vdc = vdc;
    _idq_ref_centre = idq_ref_centre;
    _gain1 = gain1;
    _gain2 = gain2;
    _gain3 = gain3;
    _cost_term_power = cost_term_power;
    _starting_mu = starting_mu;
    _mu_dec = mu_dec;
    _mu_inc = mu_inc;
    _mu_max = mu_max;
    _mu_min = mu_min;
	_dropout_max_extra_cost = dropout_max_extra_cost;
    _idq_start_positions = idq_start_positions;
    _add_dropout = add_dropout;
    _max_drop = max_drop;
    _a = a;
    _b = b;
    _starting_w1 = starting_w1;
    _starting_w2 = starting_w2;
    _starting_w3 = starting_w3;
    _w1 = starting_w1;
    _w2 = starting_w2;
    _w3 = starting_w3;

    _vdq = colvec({ vd, 0.0 });
    _vmax = vdc * sqrt(3.0 / 2.0) / 2.0;
    _mu=1.0;

    _mu_mat = mat(1, 1);
    _j_total_mat = mat(1, 1);

    _num_weights = _starting_w1.size() + _starting_w2.size() + _starting_w3.size();

    util::load_matrix_from_csv(_preloaded_idq_refs, "total_idq_refs.csv");

    // The master needs to know which samples are handled by each worker
    int rolling_sum = 0;
    for (int i = 0; i < active_workers-1; i++)
    {
        rolling_sum += num_elems_to_process();
        _sender_indices[i+1] = rolling_sum;
    }

    bool _final_cost_without_dropout_reached = false;
    double _final_cost_without_dropout = 15;
}

// ==================================================================
pair<int, int> RNN::end_start_elems_to_process()
{
    int last_idx = (_rank + 1) * _num_samples / _active_workers;
    int first_idx = _rank * _num_samples / _active_workers;
    return { first_idx, last_idx };
}

// ==================================================================
int RNN::num_elems_to_process()
{
    if (_rank >= _num_samples)
    {
        return 0;
    }

    pair<int, int> end_start = end_start_elems_to_process();
    return end_start.second - end_start.first;
}

// ==================================================================
void RNN::send_matrices(vector<mat*> matrices, vector<int>& receivers, int tag)
{
    // We don't want to make multiple calls to send multiple matrices and we don't want to convert multiple matrices
    // multiple times to arrays. So we have this function to send multiple matrices to multiple senders to consolidate work
    // The caller just packages the pointers to the matrices into a vector
    // Then, this function iterates over those matrices to figure out how the total count of the data to send
    // Then we can know how to call one MPI_Send() rather than N sends for N matrices

    int total_count = util::count_in_matrices(matrices);
    double* payload = new double[total_count];
    util::mats_to_std_array(matrices, payload);
    for (int i = 0; i < receivers.size(); i++)
    {
        MPI_Send(payload, total_count, MPI_DOUBLE, receivers[i], tag, MPI_COMM_WORLD);
    }

    delete[] payload;
}

// ==================================================================
void RNN::send_matrices(vector<mat*> matrices, int receiver, int tag)
{
    vector<int> receivers = { receiver };
    send_matrices(matrices, receivers, tag);
}

// ==================================================================
MPI_Status RNN::recv_matrices(vector<mat*> matrices, int sender, int tag)
{
    // Like with send_matrices, the idea is to provide a vector of matrices
    // (the size of each matrix determined by the caller) and then we just fill those matrices
    // with the values from the sender
    int total_count = util::count_in_matrices(matrices);
    double* payload = new double[total_count];

    MPI_Status status;
    MPI_Recv(payload, total_count, MPI_DOUBLE, sender, tag, MPI_COMM_WORLD, &status);

    util::std_array_to_mats(payload, matrices);
    delete[] payload;
    return status;
}


// ======================================================================
// Implements calculateIdq_ref.m
// The point of this function is to generate a random reference, i.e. target value for training
// We are currently preloadding these values generated from Matlab for ease of comparison
colvec RNN::calculate_idq_ref(
    const int& trajectory_number,
    const int& time_step,
    const double& ts)
{
    // NOTE: trajectory_number and time_step come from zero-indexed, rather than one-indexed code
    // So when comparing to matlab, we must add an extra value
    float period = 0.1;
    int change_number = floor(util::round_to_decimal(time_step * ts / period, 2)) + 1;

    // Well the seed computation is inconsistent.
    colvec idq_ref;
    int seed = ((trajectory_number + 1) % 10) * 10000 + change_number;
    srand(seed);
    arma_rng::set_seed(seed);
    idq_ref = colvec({ -300, -300 }) + colvec({ 600, 360 }) % mat(2, 1, arma::fill::randu);
    idq_ref = arma::round(idq_ref * 10) / 10;

    // FIXME: The calculation for the seed is all fucked, so just use the values from matlab
    idq_ref = _preloaded_idq_refs.rows(time_step * 2, time_step * 2 + 1).col(trajectory_number);
    return idq_ref;
}

// ======================================================================
mat RNN::exdiag(const mat& x)
{
    mat y = mat(x.n_rows, x.n_elem, arma::fill::zeros);
    for (uword i = 0; i < x.n_rows; i++)
    {
        y.row(i).cols(x.n_cols * i, x.n_cols * (i + 1) - 1) = x.row(i);
    }
    return y;
}

// ======================================================================
// Implements netaction.m
// The main point of this function is to compute the final output of the neural network, o3
// It also computes dnet_dw, which is the gradient (part of the FATT algorithm) 
void RNN::net_action(
    const colvec& idq,
    const colvec& idq_ref,
    const mat& hist_err,
    const mat& w1,
    const mat& w2,
    const mat& w3,
    const bool flag,
    mat& o3,
    mat& dnet_dw,
    mat& dnet_didq,
    mat& dnet_dhist_err
)
{
    colvec input0A = (idq / _gain1);
    colvec output0A = input0A.transform([](double val) { return tanh(val); });
    colvec input0B = (idq.rows(0, 1) - idq_ref) / _gain2;
    colvec output0B = input0B.transform([](double val) { return tanh(val); });
    colvec input0C = hist_err / _gain3;
    colvec output0C = input0C.transform([](double val) { return tanh(val); });

    colvec input1 = _use_idq ? join_vert(output0A, output0B, output0C, colvec({ -1 })) : join_vert(output0B, output0C, colvec({ -1 }));

    colvec sum1 = w1 * input1;
    colvec o1 = sum1.transform([](double val) { return tanh(val); });

    colvec input2 = _use_shortcuts ? join_vert(o1, input1) : join_vert(o1, colvec({ -1 }));
    colvec sum2 = w2 * input2;
    colvec o2 = sum2.transform([](double val) { return tanh(val); });

    colvec input3 = _use_shortcuts ? join_vert(o2, input2) : join_vert(o2, colvec({ -1 }));
    colvec sum3 = w3 * input3;

    o3 = sum3.transform([](double val) { return tanh(val); });

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

        if (_use_shortcuts)
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

        if (_use_idq)
        {
            dinput1_0A_0B_didq = join_vert(
                diagmat((1 - output0A % output0A) / _gain1),
                diagmat((1 - output0B % output0B) / _gain2),
                mat(2, 4, arma::fill::zeros)
            );
        }
        else
        {
            dinput1_0A_0B_didq = diagmat((1 - output0B % output0B) / _gain2);
        }
        do1_dinput1_0A_0B = diagmat(1 - o1 % o1) * w1.cols(0, w1.n_cols - 4);

        // compute Dnet_Didq
        if (_use_shortcuts)
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
        dinput1_0C_dhist_err = diagmat((1 - output0C % output0C) / _gain3);
        do1_dinput1_0C = diagmat(1 - o1 % o1) * w1.cols(w1.n_cols - 3, w1.n_cols - 2);

        if (_use_shortcuts)
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
// Main point of this algorithm is to compute the total cost of the neural network on the training data (j_total) for a each sample
// It also computes the current jacobian matrix J_total, which is used in computing the update vector, delta, in the LM algorithm
void RNN::unroll_trajectory_full(
    const colvec& initial_idq,
    const int& trajectory_number,
    const int& trajectory_length,
    const mat& w3,
    const mat& w2,
    const mat& w1,
    double& j_total,
    rowvec& e_hist_err,
    mat& j_matrix,
    mat& idq_his,
    mat& idq_ref_his)
{

    colvec idq = initial_idq;

    idq_his = mat(6, trajectory_length, arma::fill::zeros);
    idq_ref_his = mat(2, trajectory_length, arma::fill::zeros);
    mat hist_err = mat(2, trajectory_length, arma::fill::zeros);
    e_hist_err = rowvec(trajectory_length, arma::fill::zeros);

    mat didq_dw, dvdq_dw, didq_dw_matrix_sum;
    didq_dw = mat(6, _num_weights, arma::fill::zeros);
    dvdq_dw = mat(2, _num_weights, arma::fill::zeros);
    j_matrix = mat(trajectory_length + 1, _num_weights, arma::fill::zeros);
    didq_dw_matrix_sum = mat(2, _num_weights, arma::fill::zeros);

    mat err_integral, dudq_dw;
    colvec idq_ref, idq_refi;

    // outputs of net_action
    mat o3, udq, ndq, dnet_dw, dnet_didq, dnet_dhist_err;
    for (int i = 0; i < trajectory_length; i++)
    {
        err_integral = _ts * (arma::sum(hist_err, 1) - hist_err.col(i) / 2.0);
        idq_ref = calculate_idq_ref(trajectory_number, i, _ts);
        idq_ref_his.col(i) = idq_ref;
        hist_err.col(i) = (idq.rows(0, 1) - idq_ref); // when the error is too small, the calculation becomes inaccurate
        e_hist_err.col(i) = pow((arma::sum(hist_err.col(i) % hist_err.col(i))), _cost_term_power / 2.0);// the calculation process of pow is slightly from that in Matlab

        net_action(idq, idq_ref, err_integral, w1, w2, w3, false, ndq, dnet_dw, dnet_didq, dnet_dhist_err);
        udq = ndq * _vmax;

        net_action(idq, idq_ref, err_integral, w1, w2, w3, true, o3, dnet_dw, dnet_didq, dnet_dhist_err);

        if (_use_idq)
        {
            dudq_dw = _vmax * (dnet_dw + dnet_didq * didq_dw + dnet_dhist_err * _ts * didq_dw_matrix_sum);
        }
        else
        {
            dudq_dw = _vmax * (dnet_dw + dnet_didq * didq_dw.rows(0, 1) + dnet_dhist_err * _ts * (didq_dw_matrix_sum - didq_dw.rows(0, 1) / 2.0));
        }

        didq_dw = _a * didq_dw + _b * join_vert(dvdq_dw, dudq_dw, dvdq_dw);
        didq_dw_matrix_sum = didq_dw_matrix_sum + didq_dw.rows(0, 1);
        idq = _a * idq + _b * join_vert(_vdq, udq, colvec({ 0, 0 }));

        // add saturation to dq currents
        // idq(1:5) = max(min(idq(1:4), 1000*ones(4,1)),-1000*ones(4,1));
        idq.rows(0, 3) = arma::max(arma::min(idq.rows(0, 3), 1000 * mat(4, 1, arma::fill::ones)), -1000 * mat(4, 1, arma::fill::ones));

        idq_his.col(i) = idq;
        idq_refi = calculate_idq_ref(trajectory_number, i + 1, _ts);
        j_matrix.row(i + 1) = (idq.rows(0, 1) - idq_refi).t() * didq_dw.rows(0, 1) * _cost_term_power * pow(arma::sum((idq.rows(0, 1) - idq_refi) % (idq.rows(0, 1) - idq_refi)), _cost_term_power / 2.0 - 1);
    }

    j_total = arma::sum(e_hist_err % e_hist_err);
    j_matrix.shed_row(j_matrix.n_rows - 1);
}

//// ============================================================================
//void RNN::remove_smallest_weight(mat& W, vector<pair<int, int>>& dropped)
//{
//    pair<int, int> index = { 0, 0 };
//    double smallest_weight = abs(W(index.first, index.second));
//    double candidate_weight = smallest_weight;
//
//    for (int r = 0; r < W.n_rows; r++)
//    {
//        // Ignore the last column, since it's used for weights
//        for (int c = 0; c < W.n_cols - 1; c++)
//        {
//            candidate_weight = abs(W(r, c));
//
//            if (candidate_weight > 0.0 && candidate_weight < smallest_weight)
//            {
//                index.first = r;
//                index.second = c;
//                smallest_weight = candidate_weight;
//            }
//        }
//    }
//    W(index.first, index.second) = 0.0;
//    dropped.push_back(index);
//}

// ============================================================================
void RNN::find_smallest_weight(const mat& W, const vector<pair<int, int>>& dropped, double& smallest, pair<int, int>& index)
{

    //// Caller uses negative index to indicate we don't yet have an index to compare to
    //// In which case, we need to make sure to set smallest_weight to the first acceptable value in W
    //// FIXME: I hate this, but whatever
    //if (index.first == -1 && index.second == -1)
    //{
    //    cout << "Index is negative..." << endl;
	//	// We start by selecting the first index from W that isn't in dropped
	//	int row_counter = 0;
	//	int col_counter = 0;
	//	bool found_acceptable_index = false;
	//	while (!found_acceptable_index && row_counter <W.n_rows)
	//	{
	//		while (!found_acceptable_index && col_counter < W.n_cols)
	//		{
	//			// we assume we've found an acceptable index unless we find the index in dropped, in which case we move on to the next
	//			found_acceptable_index = true;

	//			index.first = row_counter;
	//			index.second = col_counter;

	//			bool found_index = false;
	//			for (int i = 0; i < dropped.size(); i++)
	//			{
	//				if (dropped[i].first == index.first && dropped[i].second == index.second)
	//				{
	//					found_acceptable_index = false;
	//					break;
	//				}
	//			}
	//			col_counter += 1;
	//		}
	//		row_counter += 1;
	//	}
    //    // This becomes our current smallest for purposes of comparison
    //    cout << "choosing this index as curent smallest weight: " << index.first << ", " << index.second << endl;
    //    current_smallest_weight = abs(W(index.first, index.second));
    //}

    smallest = -1;
    for (int r = 0; r < W.n_rows; r++)
    {
        // Ignore the last column, since it's used for weights
        for (int c = 0; c < W.n_cols - 1; c++)
        {
            bool valid_candidate = true;
			for (int i = 0; i < dropped.size(); i++)
			{
				if (dropped[i].first == r && dropped[i].second == c )
				{
                    cout << "Found this index already in dropped, so skipping..." << r << ", " << c << endl;
                    valid_candidate = false;
                    break;
				}
			}
            if (valid_candidate)
            {
				double candidate_weight = abs(W(r, c));
				if (smallest == -1 || candidate_weight < smallest)
				{
					index.first = r;
					index.second = c;
					smallest = candidate_weight;
				}
            }
        }
    }
}

// ============================================================================
void RNN::remove_smallest_weight(mat& w1, mat& w2, mat& w3)
{

    bool update_w1, update_w2, update_w3 = false;

    // We need to make sure our first selected value is not already in w1_dropped
    pair<int, int> w1_index, w2_index, w3_index;
    double w1_smallest, w2_smallest, w3_smallest;

    find_smallest_weight(w1, _w1_dropped, w1_smallest, w1_index);
    find_smallest_weight(w2, _w2_dropped, w2_smallest, w2_index);
    find_smallest_weight(w3, _w3_dropped, w3_smallest, w3_index);

    if (w1_smallest < w2_smallest && w1_smallest < w3_smallest)
    {
        cout << "Removing weight from w1, indices: " << w1_index.first << ", " << w1_index.second << ", value: " << w1(w1_index.first, w1_index.second) << endl;
        _last_weight_dropped = w1(w1_index.first, w1_index.second);
		w1(w1_index.first, w1_index.second) = 0.0;
		_w1_dropped.push_back(w1_index);
    }
    else if (w2_smallest < w1_smallest && w2_smallest < w3_smallest)
    {
        cout << "Removing weight from w2, indices: " << w2_index.first << ", " << w2_index.second << ", value: " << w2(w2_index.first, w2_index.second) << endl;
        _last_weight_dropped = w2(w2_index.first, w2_index.second);
		w2(w2_index.first, w2_index.second) = 0.0;
		_w2_dropped.push_back(w2_index);
    }

    else if (w3_smallest < w1_smallest && w3_smallest < w2_smallest)
    {
        cout << "Removing weight from w3, indices: " << w3_index.first << ", " << w3_index.second << ", value: " << w3(w3_index.first, w3_index.second) << endl;
        _last_weight_dropped = w3(w3_index.first, w3_index.second);
		w3(w3_index.first, w3_index.second) = 0.0;
		_w3_dropped.push_back(w3_index);
    }
}

// ==================================================================
void RNN::apply_dropout(mat& W, vector<pair<int, int>>& dropped)
{
    pair<int, int> to_drop;
    for (int i = 0; i < dropped.size(); i++)
    {
        to_drop = dropped.at(i);
        W(to_drop.first, to_drop.second) = 0.0;
    }
    return;
}

// ============================================================================
void RNN::train_best_weights(const int max_iterations, bool verbose)
{
    // each worker computes how many elements it needs to process    
    const int samples_to_process = num_elems_to_process();

	// Some workers may have nothing to do
	if (samples_to_process == 0) { return; }
	
	// Each worker computes its reserved indices for depositing values into the aggregation variables
	const pair<int, int> range_of_samples = end_start_elems_to_process();

	double mu = _starting_mu;
    int current_drop = _w1_dropped.size() + _w2_dropped.size() + _w3_dropped.size();

    colvec idq, dw, dw_y, rr;
    mat j_matrix, j_matrix2;
    rowvec e_hist_err, e_hist_err2;
    mat jj, ii, h_matrix;
    double j_total, j_total_subsum, j_total_sum;
    double j_total2, j_total_subsum2, j_total_sum2;
    vector<mat*> matrices;

    rowvec hist_err_top_ten;
    mat j_matrix_top_ten;

    mat w1_temp = mat(_w1.n_rows, _w1.n_cols);
    mat w2_temp = mat(_w2.n_rows, _w2.n_cols);
    mat w3_temp = mat(_w3.n_rows, _w3.n_cols);
    mat mu_mat = mat(1, 1);
    mat j_total_mat = mat(1, 1);

    bool success;
    MPI_Status status;

	j_total_sum = 0;
	j_total_sum2 = 0;
	for (int iteration = 1; iteration < max_iterations + 1; iteration++)
	{
		// Use FATT to calculate total cost of each trajectory, the error vector, and the jacobian matrix. 
		mat idq_his, idq_ref_his;

		rowvec hist_err_subtotal = rowvec(_trajectory_length * samples_to_process, arma::fill::zeros);
		mat j_matrix_subtotal = mat(_trajectory_length * samples_to_process, _num_weights, arma::fill::zeros);
		j_total_subsum = 0;

		// Each worker (master included) does its own part for this loop
		for (int i = range_of_samples.first, j = 0; i < range_of_samples.second; i++, j++)
		{
			idq = _idq_start_positions.col(i);
			unroll_trajectory_full(idq, i, _trajectory_length, _w3, _w2, _w1, j_total, e_hist_err, j_matrix, idq_his, idq_ref_his);

			// FOR TESTING, we keep only the top ten values so they are al the same for purposes of getting same training results across higher trajectories
			if (i < 10)
			{
				j_total_subsum += j_total;
			}
			hist_err_subtotal.cols(j * (_trajectory_length), (j + 1) * (_trajectory_length)-1) = e_hist_err;
			j_matrix_subtotal.rows(j * (_trajectory_length), (j + 1) * (_trajectory_length)-1) = j_matrix;
		}

		// Master will aggregate results from workers
		if (_rank == 0)
		{
			rowvec hist_err_total = rowvec(_trajectory_length * _num_samples, arma::fill::zeros);
			mat j_matrix_total = mat(_trajectory_length * _num_samples, _num_weights, arma::fill::zeros);
			j_total_sum = 0;

			// The master first adds its piece to the aggregation variables, then waits to get chunks from the workers
			j_total_sum = j_total_subsum;
			hist_err_total.cols(0, samples_to_process * _trajectory_length - 1) = hist_err_subtotal;
            j_matrix_total.rows(0, samples_to_process * _trajectory_length - 1) = j_matrix_subtotal;

            for (int senderid=1; senderid < _active_workers; senderid++)
			{
				int sender_samples = num_elems_to_process();

				// we receive these subtotals from sender
				hist_err_subtotal = rowvec(_trajectory_length * sender_samples, arma::fill::zeros);
				j_matrix_subtotal = mat(_trajectory_length * sender_samples, _num_weights, arma::fill::zeros);
				j_total_mat.zeros();

				status = recv_matrices({ &j_total_mat, &hist_err_subtotal, &j_matrix_subtotal }, senderid);

				int start = _sender_indices[senderid] * _trajectory_length;
				int end = start + (sender_samples * _trajectory_length) - 1;

				// and update our totals
				hist_err_total.cols(start, end) = hist_err_subtotal;
				j_matrix_total.rows(start, end) = j_matrix_subtotal;
				j_total_sum += j_total_mat.at(0, 0);

			}
            j_total_sum = j_total_sum;

			// FIXME: FOR TESTING, we are only using the values computed for j_total, j_matrix_total, and hist_err_total for the top ten
			// samples, so our convergence should be identical for all trajectories we are testing for performance reasons 

			hist_err_top_ten = rowvec(_trajectory_length * 10, arma::fill::zeros);
			j_matrix_top_ten = mat(_trajectory_length * 10, _num_weights, arma::fill::zeros);

			for (int i = 0; i < std::min(10, _num_samples); i++)
			{
				int start = i * (_trajectory_length - 1);
				int end = (i + 1) * (_trajectory_length - 1) - 2;

				// In the original matlab, we skip the first value for some reason. Don't know why. Ugly as sin
				hist_err_top_ten.cols(start, end) = hist_err_total.cols((_trajectory_length * i) + 1, _trajectory_length * (i + 1) - 2);
				j_matrix_top_ten.rows(start, end) = j_matrix_total.rows((_trajectory_length * i) + 1, _trajectory_length * (i + 1) - 2);
			}

			// Now that we've computed j_total, hist_err_total, and j_matrix_total, 
			// master can do cholensky decomposition to solve for weight updates
			mat L = mat(_num_weights, _num_weights, arma::fill::zeros);
			mat R = mat(_num_weights, _num_weights, arma::fill::zeros);
			while (mu < _mu_max)
			{
				jj = j_matrix_top_ten.t() * j_matrix_top_ten;
				ii = -1 * j_matrix_top_ten.t() * arma::vectorise(hist_err_top_ten);
				h_matrix = jj + mu * arma::eye(_num_weights, _num_weights);

				// cholensky decomposition to solve for weight updates
				// FIXME: Armadillo chol just doesn't work? Use Eigen instead
				success = util::cholesky_eigen(L, R, h_matrix);

				while (!success)
				{
					mu = mu * _mu_inc;
					if (mu == _mu_max)
					{
						break;
					}
					h_matrix = jj + mu * arma::eye(_num_weights, _num_weights);
					success = util::cholesky_eigen(L, R, h_matrix);
				}
				if (mu == _mu_max)
				{
					break;
				}
				arma::solve(dw_y, L, ii) && arma::solve(dw, R, dw_y);

				w3_temp = _w3 + arma::reshape(dw.rows(0, _w3.n_elem), _w3.n_cols, _w3.n_rows).t();
				w2_temp = _w2 + arma::reshape(dw.rows(_w3.n_elem, _w3.n_elem + _w2.n_elem), _w2.n_cols, _w2.n_rows).t();
				w1_temp = _w1 + arma::reshape(dw.rows(_w3.n_elem + _w2.n_elem, dw.n_elem - 1), _w1.n_cols, _w1.n_rows).t();

                // Because we compute the candidate weights by adding the values computed using cholesky decomp,
                // we need to make sure to zero out those weights every time. So we track the weights we've
                // chosen to drop using w1_dropped, w2_dropped, w3_dropped
                if (_add_dropout)
                {
                    apply_dropout(w1_temp, _w1_dropped);
                    apply_dropout(w2_temp, _w2_dropped);
                    apply_dropout(w3_temp, _w3_dropped);
                }

				// As soon as we've computed these new weights, we can give them to the workers to do their pieces
				send_matrices({ &w1_temp, &w2_temp, &w3_temp }, _workerids, MPI_NEW_JOB);

				// Master needs to compute its piece of the sum
				j_total_sum2 = 0;
				j_total2 = 0;

				for (int i = range_of_samples.first; i < range_of_samples.second; i++)
				{
					idq = _idq_start_positions.col(i);
					unroll_trajectory_full(idq, i, _trajectory_length, w3_temp, w2_temp, w1_temp, j_total2, e_hist_err2, j_matrix2, idq_his, idq_ref_his);

                    // FIXME: Just so we compute the same cost regardless of trajectories for purpose of testing
					if (i < 10)
					{
						j_total_sum2 += j_total2;
					}
				}

				// Master needs to get sums from the workers
				for (int i = 1; i < _active_workers; i++)
				{

					j_total_subsum2 = 0;
					MPI_Recv(&j_total_subsum2, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_J_TOTAL, MPI_COMM_WORLD, &status);
					j_total_sum2 += j_total_subsum2;
				}

				if (j_total_sum2 < j_total_sum)
				{
                    j_total_sum = j_total_sum2;
                    // FIXME: I think this is the problem. What happpens when w1_temp and so on go out of scope? COme back to this...
					_w3 = w3_temp;
					_w2 = w2_temp;
					_w1 = w1_temp;

					rr = join_cols(rr, colvec(j_total_sum2 / _trajectory_length / _num_samples));
					mu = std::max(mu * _mu_dec, _mu_min);

                    if (verbose)
                    {
						std::cout << setprecision(16)
							<< "iteration: " << iteration
							<< ", mu=" << mu
							<< ", J_total_sum2=" << j_total_sum2 / _trajectory_length / 10 << endl;
                    }
					break;
				}

				mu = mu * _mu_inc;

			} // while mu < mu_max loop ends here

			// Anytime we break out of this loop, we need to let our workers know
			mu_mat = mat({ mu });
			send_matrices({ &_w1, &_w2, &_w3, &mu_mat }, _workerids, MPI_LOOP_DONE);
		}

		// The workers have done their work, so they just send to the master and wait for the aggregation
		else
		{
			// Then we send the matrices
			j_total_mat.at(0, 0) = j_total_subsum;

			send_matrices({ &j_total_mat, &hist_err_subtotal, &j_matrix_subtotal }, 0, MPI_JOB_FINISHED);

			// Each worker will stay in this loop until master has decided we are done updating weights and mu
			while (true)
			{
				status = recv_matrices({ &w1_temp, &w2_temp, &w3_temp, &mu_mat }, 0);
				if (status.MPI_TAG == MPI_NEW_JOB)
				{
					j_total_subsum2 = 0;
					for (int i = range_of_samples.first; i < range_of_samples.second; i++)
					{
						idq = _idq_start_positions.col(i);
						unroll_trajectory_full(idq, i, _trajectory_length, w3_temp, w2_temp, w1_temp, j_total2, e_hist_err2, j_matrix2, idq_his, idq_ref_his );

                        // FIXME: Just so we compute the same cost regardless of trajectories for purpose of testing
						if (i < 10)
						{
							j_total_subsum2 += j_total2;
						}
					}
					MPI_Send(&j_total_subsum2, 1, MPI_DOUBLE, 0, MPI_J_TOTAL, MPI_COMM_WORLD);
				}
				else if (status.MPI_TAG == MPI_LOOP_DONE)
				{
					_w3 = w3_temp;
					_w2 = w2_temp;
					_w1 = w1_temp;
					mu = mu_mat.at(0, 0);
					break;
				}
			}
		}

		if (mu == _mu_max)
		{
            // FIXME: This doesn't work with parallelization right now...
            if (_add_dropout)
            {
                if (!_final_cost_without_dropout_reached)
                {
                    _final_cost_without_dropout = j_total_sum / _trajectory_length / 10;
                    _final_cost_without_dropout_reached = true;
                    cout << "Final cost without dropout: " << _final_cost_without_dropout << endl;
                }

                // We only quit training if the current cost is greater than the final_cost_without_dropout + dropout_max_extra_cost
                double current_cost = j_total_sum / _trajectory_length / 10;
                cout << "Current cost: " << current_cost << endl;

                if (current_drop < _max_drop)
                {
                    //if (current_cost < _final_cost_without_dropout + _dropout_max_extra_cost)
                    //{
						cout << "Removing weight when max_drop is " << _max_drop << endl;

						// Then we continue training until max iterations, after removing one weight at a time
						remove_smallest_weight(_w1, _w2, _w3);
						current_drop++;

						// We have to reset mu since necessarily it will have been maxed at this point
						// mu = _starting_mu;
						continue;
                    //}

                    // FIXME: Need to figure out if we should keep the weights *prior* to this bad cost value?
                    //cout << "Cost is too high, ending training with " << current_drop << " weights dropped..." << endl;
                }
            }
			break;
		}
	}
}

// ============================================================================
void RNN::set_weights(const mat& w1, const mat& w2, const mat& w3)
{
    _w1 = w1;
    _w2 = w2;
    _w3 = w3;
}

//// ============================================================================
//void RNN::set_dropout(const bool add_dropout, const double dropout_max_extra_cost)
//{
//    _add_dropout = add_dropout;
//    _dropout_max_extra_cost = dropout_max_extra_cost;
//}

// ============================================================================
void RNN::get_weights(mat &w1, mat &w2, mat &w3)
{
    w1 = _w1;
    w2 = _w2;
    w3 = _w3;
}

