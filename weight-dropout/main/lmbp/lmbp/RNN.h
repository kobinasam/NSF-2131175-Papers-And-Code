#pragma once

#include <mpi.h>
#include <iomanip>
#include <armadillo>
#include "matrix_utility.h"

using namespace std;
using namespace arma;

#define _USE_MATH_DEFINES

class RNN
{

private:

	int _rank;
	vector<int> _workerids;
	int _active_workers;
	int _num_samples;
	bool _use_shortcuts;
	bool _use_idq;
	bool _add_dropout;
	double _vd;
	double _t_final;
	double _ts;
	double _vdc;
	colvec _idq_ref_centre;
	double _gain1;
	double _gain2;
	double _gain3;
	double _cost_term_power;
	double _starting_mu;
	double _mu_dec;
	double _mu_inc;
	double _mu_max;
	double _mu_min;
	double _dropout_max_extra_cost;
	int _max_drop;
	mat _idq_start_positions;
	mat _a;
	mat _b;
	mat _w1;
	mat _w2;
	mat _w3;
	mat _starting_w1;
	mat _starting_w2;
	mat _starting_w3;

    colvec _vdq;
    double _vmax;

    double _mu;
	int _trajectory_length;

    mat _mu_mat;
    mat _j_total_mat;
    mat _preloaded_idq_refs;

    int _num_weights;

	map<int, int> _sender_indices;

	vector<pair<int, int>> _w1_dropped, _w2_dropped, _w3_dropped;
	vector<pair<int, int>> _w1_candidates, _w2_candidates, _w3_candidates;

	// Kind of a dumb variable and only used for testing
	double _last_weight_dropped;

protected:

	pair<int, int> end_start_elems_to_process();
	int num_elems_to_process();

	void send_matrices(vector<mat*> matrices, vector<int>& receivers, int tag);
	void send_matrices(vector<mat*> matrices, int receiver, int tag);
	MPI_Status recv_matrices(vector<mat*> matrices, int sender, int tag = MPI_ANY_TAG);

	colvec calculate_idq_ref(const int& trajectory_number, const int& time_step, const double& ts);
	mat exdiag(const mat& x);

	void net_action(
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
	);

	void remove_smallest_weight(mat& w1, mat& w2, mat& w3);
	void remove_smallest_weight(mat& W, pair<int, int> index, vector<pair<int, int>>& dropped, vector<pair<int, int>>& candidates);
	void find_smallest_weight(const mat& W, const vector<pair<int, int>>& candidates, double& smallest, pair<int, int>& index);
	void apply_dropout(mat& W, vector<pair<int, int>>& dropped);
	void initialize_candidates(const mat& M, vector<pair<int, int>>& candidates);

public:

	// ------------------------------------------------------------------------
	// getters
	double get_ts() { return _ts; }
	double get_trajectory_length() { return _trajectory_length; }
	void get_weights(mat &w1, mat& w2, mat &w3);
	double get_last_weight_dropped() { return _last_weight_dropped; }

	// ------------------------------------------------------------------------
	// setters
	void set_weights(const mat& w1, const mat& w2, const mat& w3);
	void set_dropout(const bool add_dropout) { _add_dropout = add_dropout; }
	void set_max_drop(const int max_drop) { _max_drop = max_drop; }

	void clear_weights();
	void train_best_weights(const int max_iterations, bool verbose);

	void unroll_trajectory_full(
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
		mat& idq_ref_his);
		
		// class RandomNumberGenerator1 {
		// public:
		// 	static arma::mat generateRandomNumbers() {
		// 		arma::arma_rng::set_seed(std::time(nullptr));
		// 		return arma::randu<arma::mat>(6,5);
		// 	}
		// };
		// class RandomNumberGenerator2 {
		// public:
		// 	static arma::mat generateRandomNumbers() {
		// 		arma::arma_rng::set_seed(std::time(nullptr));
		// 		return arma::randu<arma::mat>(6,7);
		// 	}
		// };
		// class RandomNumberGenerator3 {
		// public:
		// 	static arma::mat generateRandomNumbers() {
		// 		arma::arma_rng::set_seed(std::time(nullptr));
		// 		return arma::randu<arma::mat>(2,7);
		// 	}
		// };


	RNN(
		int rank,
		vector<int> workerids, 
		int active_workers, 
		int num_samples,
		const mat idq_start_positions,
		const bool add_dropout = false,
		const double dropout_max_extra_cost = 0.0,
		const int max_drop = 0,
		const bool use_shortcuts = false,
		const bool use_idq = false,
		const double vd = 690,
		const double t_final = 1,
		const double ts = 0.001,
		const double vdc = 1200,
		const colvec idq_ref_centre = colvec({ 0, 0 }),
		const double gain1 = 1000.0,
		const double gain2 = 100.0,
		const double gain3 = 100.0,
		const double cost_term_power = 1.0 / 2.0,
		const double starting_mu = 1.0,
		const double mu_dec = 0.1,
		const double mu_inc = 10,
		const double mu_max = 1e10,
		const double mu_min = 1e-20,
		const mat a = {
			{ 0.922902679404235, 0.365403020170600, 0.001311850123628, 0.000519398207289, -0.006031602076712, -0.002388080200093},
			{ -0.365403020170600, 0.922902679404235, -0.000519398207289, 0.001311850123628, 0.002388080200093, -0.006031602076712},
			{ 0.001311850123628, 0.000519398207289, 0.922902679404235, 0.365403020170600, 0.006031602076712, 0.002388080200093},
			{ -0.000519398207289, 0.001311850123628, -0.365403020170600, 0.922902679404235, -0.002388080200093, 0.006031602076712},
			{ 0.120632041534246, 0.047761604001858, -0.120632041534246, -0.047761604001858, 0.921566702872299, 0.364874069642510},
			{ -0.047761604001858, 0.120632041534245, 0.047761604001858, -0.120632041534245, -0.364874069642510, 0.921566702872299}
		},

		const mat b = {
			{0.488106762997528, 0.093547911260568, -0.485485431756243, -0.091984091707451, -0.000001945097416, 0.000009097619657 },
			{-0.093547911260568, 0.488106762997528, 0.091984091707451, -0.485485431756243, -0.000009097619657, -0.000001945097416 },
			{0.485485431756243, 0.091984091707451, -0.488106762997528, -0.093547911260568, 0.000001945097416, -0.000009097619657 },
			{-0.091984091707451, 0.485485431756243, 0.093547911260568, -0.488106762997528, 0.000009097619657, 0.000001945097416 },
			{0.038901948324797, -0.181952393142100, 0.038901948324797, -0.181952393142100, 0.000002613550852, 0.000001600210032 },
			{0.181952393142100, 0.038901948324797, 0.181952393142100, 0.038901948324797, -0.000001600210032, 0.000002613550852 }
		},

		mat starting_w1 = {
			{6.3129816971085571e-03,2.5266590369958605e-03,2.2876219888541293e-03,1.8201608129544934e-03,6.8808085409051336e-03},
			{3.2677360678404835e-03,8.6213260574343007e-04,7.0333372301132069e-03,1.4536184727664309e-03,2.0057786465571413e-03},
			{1.9849465319988799e-03,5.7045323826978205e-03,7.8625996561314353e-03,1.0212898775492868e-03,2.0939173714513985e-03},
			{9.3908465432710076e-03,6.4411057924055630e-03,2.9212062389581358e-03,2.8457624007679197e-03,4.8188629742971249e-03},
			{1.4140062795952710e-03,8.3451101667531935e-04,8.1830392762490111e-03,9.5385623627910286e-03,5.5448952548592233e-03},
			{8.6433766471117322e-04,6.5071752357817884e-04,5.9404502211526109e-03,3.4811497130793081e-03,5.6737226093434272e-03}
		},

		mat starting_w2 = {
			{3.9593268430865051e-03,6.2583509078695957e-03,6.5937510016904214e-03,5.7215169316423078e-03,3.4944952615449695e-03,9.9178092009344562e-03,1.4195026451172196e-03},
			{2.1468787694930514e-03,1.2778282065839424e-03,3.5034240116104671e-03,6.0999231269730789e-03,3.9787206452355974e-04,7.1535839469804088e-03,8.5365136091737539e-03},
			{4.2415743403470870e-03,9.7674133766270915e-04,3.9493516277653722e-04,5.0053761666241764e-04,3.7014537971542887e-03,4.7298779064942368e-03,2.4559826194268937e-03},
			{5.7589690486073977e-03,6.4020668345261753e-03,4.1580044178911139e-03,3.2784959697193445e-03,6.9931150666165702e-03,6.4733788207062548e-03,3.3431158802691091e-03},
			{3.7013409523404747e-03,7.8113278971844316e-03,9.0056478556489598e-04,1.6794083105804258e-04,6.7207120188701990e-03,6.6394922081557508e-03,2.4551490633832572e-03},
			{7.8430428046821089e-03,5.7077629407387403e-03,4.9746523094853033e-03,5.5440760877656335e-03,5.0035885946540862e-06,1.2454351873828545e-03,2.5629350292230113e-04}
		},

		mat starting_w3 = {
			{8.3682912591092427e-03,1.1224509383133631e-03,5.1347677686851209e-03,2.7204780399745741e-03,7.0529512776038115e-04,4.2567591625703345e-03,2.6570581322535908e-03},
			{9.3206915925580884e-03,9.5409750302816801e-03,8.9787575066743158e-03,8.6087499411677103e-03,8.1022939369870921e-03,5.8024043745788139e-03,5.3826842122386802e-03}
		}
		// w1 = [5*6],
		// w2 = [7*6],
		// w3 = [7*2]
		// arma::mat starting_w1 = arma::randu(6, 5) * 0.01,
		// arma::mat starting_w2 = arma::randu(6, 7) * 0.01,
		// arma::mat starting_w3 = arma::randu(2, 7) * 0.01
	);
		  
};
