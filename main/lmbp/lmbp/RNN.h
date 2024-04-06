#pragma once

#include <iomanip>
#include <armadillo>

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
	bool _final_cost_without_dropout_reached;
    double _final_cost_without_dropout;
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

	//void remove_smallest_weight(mat& W, vector<pair<int, int>>& dropped);
	void remove_smallest_weight(mat& w1, mat& w2, mat& w3);
	void find_smallest_weight(const mat& W, const vector<pair<int, int>>& dropped, double& smallest, pair<int, int>& index);
	void apply_dropout(mat& W, vector<pair<int, int>>& dropped);

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
			{2.85180510569669e-02,9.83396060410573e-02,4.7217822137635e-02,9.36941266862299e-03,2.61944840921349e-02},
			{4.5194388250585e-02,7.74058541059018e-02,8.06265263909307e-02,7.36328462708989e-02,2.66434659897778e-02},
			{6.43638228936951e-02,2.70985951064152e-02,5.32568320591652e-02,1.55029925751549e-02,7.29687665415855e-02},
			{1.48363198371998e-02,5.61789706132784e-03,1.52273543049854e-02,5.94156921356289e-02,9.6238840874235e-02},
			{4.81506624057505e-02,4.82365066339309e-02,1.76356294407989e-02,7.11553172004113e-02,5.36133395972862e-02},
			{2.91227981042624e-02,5.00255623871338e-02,9.24749761738724e-02,6.55882481644463e-02,8.01044805040259e-02}
		},

		mat starting_w2 = {
			{9.72663346892104e-02,9.41518756985262e-02,9.24684526766533e-02,6.76028420316214e-02,3.89073963121502e-02,6.2885055218047e-02,6.17739155178077e-02},
			{6.17611098752424e-02,3.38969174864889e-02,7.56887838803959e-02,5.7392153945771e-02,8.28266889965628e-02,5.28195155446723e-02,7.6240222058896e-02},
			{2.72439892460214e-02,7.14222654707556e-02,5.58245230713163e-02,4.19832235935119e-02,3.92436308818954e-02,8.34302927490567e-02,	4.81750548446936e-02},
			{7.73237148942618e-02,4.76891126731468e-02,1.66468777982221e-02,1.56203365330868e-02,2.99953712660398e-02,7.07797242799758e-02,	8.22275761307158e-04},
			{1.27418452279345e-02,6.3559145535163e-02,2.12717625086635e-02,7.85610651394205e-02,5.31492383030328e-02,2.77713857889045e-02,	6.54204312374759e-02},
			{1.4537496715222e-02,4.70579385093234e-03,3.29450644050456e-02,7.30277724824863e-02,3.59642343455937e-03,5.6823072354395e-02,8.15529348926356e-02}
		},

		mat starting_w3 = {
			{9.56824093673647e-02,8.48666955305194e-03,8.55992697235381e-02,8.19770415594325e-02,3.52220162999883e-02,7.88074933218939e-02,1.05170047737392e-02},
			{7.99461277908286e-04,6.49912454784609e-02,4.29100222685185e-02,3.68985778323839e-02,7.01618724729998e-02,5.37753467817589e-02,1.12796849104818e-02}
		}
	);
};
