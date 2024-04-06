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
			{ 0.081472368639318, 0.027849821886705, 0.095716694824295, 0.079220732955955, 0.067873515485777 },
			{ 0.090579193707562, 0.054688151920498, 0.048537564872284, 0.095949242639290, 0.075774013057833 },
			{ 0.012698681629351, 0.095750683543430, 0.080028046888880, 0.065574069915659, 0.074313246812492 },
			{ 0.091337585613902, 0.096488853519928, 0.014188633862722, 0.003571167857419, 0.039222701953417 },
			{ 0.063235924622541, 0.015761308167755, 0.042176128262628, 0.084912930586878, 0.065547789017756 },
			{ 0.009754040499941, 0.097059278176062, 0.091573552518907, 0.093399324775755, 0.017118668781156 }
		},

		mat starting_w2 = {
			{ 0.012649986532930, 0.031747977514944, 0.055573794271939, 0.055778896675488, 0.025779225057201, 0.040218398522248, 0.087111112191539 },
			{ 0.013430330431357, 0.031642899914629, 0.018443366775765, 0.031342898993659, 0.039679931863314, 0.062067194719958, 0.035077674488589 },
			{ 0.009859409271100, 0.021756330942282, 0.021203084253232, 0.016620356290215, 0.007399476957694, 0.015436980547927, 0.068553570874754 },
			{ 0.014202724843193, 0.025104184601574, 0.007734680811268, 0.062249725927990, 0.068409606696201, 0.038134520444447, 0.029414863376785 },
			{ 0.016825129849153, 0.089292240528598, 0.091380041077957, 0.098793473495250, 0.040238833269616, 0.016113397184936, 0.053062930385689 },
			{ 0.019624892225696, 0.070322322455629, 0.070671521769693, 0.017043202305688, 0.098283520139395, 0.075811243132742, 0.083242338628518 }
		},

		mat starting_w3 = {
			{ 0.002053577465818, 0.065369988900825, 0.016351236852753, 0.079465788538875, 0.044003559576025, 0.075194639386745, 0.006418708739190 },
			{ 0.092367561262041, 0.093261357204856, 0.092109725589220, 0.057739419670665, 0.025761373671244, 0.022866948210550, 0.076732951077657 }
		}
	);
};
