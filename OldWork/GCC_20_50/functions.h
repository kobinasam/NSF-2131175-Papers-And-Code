
#ifndef MYFUNCTIONS
#define MYFUNCTIONS

#include <vector>
#include <math.h>
#include <utility>
#include <fstream>

#include <armadillo>

using namespace std;
using namespace arma;

#define _USE_MATH_DEFINES

// ==================================================================
// Utility functions start here
void print_mat(const mat m, const string &name, bool hide_contents=false, double precision=5)
{
    cout << "============================================" << endl;
    cout << "Name: " << name << ", Rows: " << m.n_rows << ", Cols: " << m.n_cols << endl;
    if (!hide_contents)
    {
        for (uword i = 0; i < m.n_rows; i++)
        {
            for (uword j = 0; j < m.n_cols; j++)
            {
                cout << setprecision(precision) << m.at(i, j) << " ";
            }
            cout << endl;
        }
    }
    if (m.is_empty())
    {
        cout << name << " is empty!";
    }
    cout << endl;
}

// ==================================================================
void save_matrix_to_csv(const mat &M, string filename)
{
    ofstream file_handle;
    file_handle.open(filename);
    if (!file_handle.good())
    {
        throw std::runtime_error("Problem writing to file: " + filename);
    }
    M.save(file_handle, arma::csv_ascii);
    file_handle.close();
}

// ==================================================================
void load_matrix_from_csv(mat &M, string filename)
{
    ifstream file_handle;
    file_handle.open(filename);
    if (!file_handle.good())
    {
        throw std::runtime_error("Filepath does not exist: " + filename);
    }
    M.load(file_handle, arma::csv_ascii);
    file_handle.close();
}

// ==================================================================
void show_largest_diff(const mat &M1, const mat &M2, string name)
{
    cout << "Largest diff in " << name << ": " << arma::abs(M1 - M2).max() << endl;
}

// ==================================================================
void show_digits_of_diff(const mat &M1, const mat &M2, string name)
{
    double tolerance = 1e-17;
    bool same = false;

    while (!same)
    {
        tolerance = tolerance * 10;
        same = approx_equal(M1, M2, "reldiff", tolerance);
    }

    cout << name << " are same within " << tolerance << " tolerance" << endl;
}

// ==================================================================
string generate_subdir(vector<std::pair<string, bool>> conditions)
{
    string subdir = "";

    for (int i = 0; i < conditions.size(); i++)
    {
        subdir += conditions[i].second ? "yes": "no";
        subdir += "_";
        subdir += conditions[i].first;
        subdir += "_";
    }
    if (!conditions.empty())
    {
        subdir.pop_back();
    }
    return subdir;
}

// ==================================================================
// Core functions start here

// ======================================================================
// Implements modifyIdq_ref.m
colvec modify_idq_ref(
    const colvec &idq,
    const colvec &vdq,
    const double &vmax,
    const double &xl)
{
    double vd = vdq[0];
    double id_ref = idq[0];
    double iq_ref = idq[1];

    double vq1 = id_ref * xl;
    double vd1 = sqrt(vmax * vmax - vq1 * vq1);

    double iq_ref_new = (vd1 - vd) / xl;
    iq_ref = iq_ref > iq_ref_new ? iq_ref_new: iq_ref;

    return colvec({id_ref, iq_ref});
}

// ======================================================================
// Implements calculateIdq_ref.m
colvec calculate_idq_ref(
    const double &trajectory_number,
    const double &time_step,
    const double &ts,
    const double &imax,
    const double &iq_max)
    // const double &iq_max,
    // const colvec &vdq,
    // const double &vmax,
    // const double &xl)
{
    colvec idq_ref = colvec({imax / 2.0, 0.0});

    // This is the time interval that specifies how often idq_ref is to change (in seconds);
    float period = 0.1;
	int change_number = (floor((time_step ) * ts / period));// not time_step-1?

    if (change_number > 0)
    {
        srand(trajectory_number*10000+change_number);
		idq_ref = colvec({ -1 * imax, -1 * imax }) + colvec({ 2 * imax, iq_max + imax }) % mat(2, 1, arma::fill::randu);
        // idq_ref = modify_idq_ref(idq_ref, {vdq[0], vdq[1]}, vmax, xl);
    }

    return idq_ref;
}

// ======================================================================
mat exdiag(const mat &x)
{
    mat y = mat(x.n_rows, x.n_elem, arma::fill::zeros);
    for (uword i = 0; i < x.n_rows; i++)
    {
        y.row(i).cols(x.n_cols * i, x.n_cols * (i+1) - 1) = x.row(i);
    }
    return y;
}

// ======================================================================
void net_action(
    const colvec &idq,
    const colvec &idq_ref,
    const mat &hist_err,
    const mat &w1,
    const mat &w2,
    const mat &w3,
    const bool flag,
    const bool use_shortcuts,
    const bool use_idq,
    const double gain1,
    const double gain2,
    const double gain3,
    mat &o3,
    mat &dnet_dw,
    mat &dnet_didq,
    mat &dnet_dhist_err
    )

{
    colvec input0A = (idq / gain1);
    colvec output0A = input0A.transform([](double val) { return tanh(val); } );
    colvec input0B = (idq - idq_ref) / gain2;
    colvec output0B = input0B.transform([](double val) { return tanh(val); } );
    colvec input0C = hist_err / gain3;
    colvec output0C = input0C.transform([](double val) { return tanh(val); } );

    colvec input1 = use_idq ? join_vert(output0A, output0B, output0C, colvec({-1})): join_vert(output0B, output0C, colvec({-1}));

    colvec sum1 = w1 * input1;
    colvec o1 = sum1.transform([](double val) { return tanh(val); } );

    colvec input2 = use_shortcuts ? join_vert(o1, input1): join_vert(o1, colvec({-1}));
    colvec sum2 = w2 * input2;
    colvec o2 = sum2.transform([](double val) { return tanh(val); } );

    colvec input3 = use_shortcuts ? join_vert(o2, input2): join_vert(o2, colvec({-1}));
    colvec sum3 = w3 * input3;
    o3 = sum3.transform([](double val) { return tanh(val); } );

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

        if (use_shortcuts)
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

        if (use_idq)
        {
            dinput1_0A_0B_didq = join_vert(
                diagmat((1-output0A % output0A) / gain1),
                diagmat((1-output0B % output0B) / gain2)
            );
        }
        else
        {
            dinput1_0A_0B_didq = diagmat((1-output0B % output0B) / gain2);
        }
        do1_dinput1_0A_0B = diagmat(1 - o1 % o1) * w1.cols(0, w1.n_cols - 4);

        // compute Dnet_Didq
        if (use_shortcuts)
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
        dinput1_0C_dhist_err = diagmat((1 - output0C % output0C) / gain3);
        do1_dinput1_0C = diagmat(1 - o1 % o1) * w1.cols(w1.n_cols - 3, w1.n_cols - 2);

        if (use_shortcuts)
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
void unroll_trajectory_full(
    const colvec &initial_idq,
	const colvec &vdq,
    const double &trajectory_number,
    const double &trajectory_length,
    const mat &w3,
    const mat &w2,
    const mat &w1,
    const bool flag,
    const bool use_shortcuts,
    const bool use_idq,
    const mat &idq_ref_total,
    const mat &a,
    const mat &b,
    const double &vmax,
    const double &ts,
    const double &gain1,
    const double &gain2,
    const double &gain3,
    const double &cost_term_power,
    double &j_total,
    rowvec &e_hist_err,
    mat &j_matrix
)
{
    colvec idq = initial_idq;
    double num_weights = w1.size() + w2.size() + w3.size();
    mat hist_err = mat(2, trajectory_length, arma::fill::zeros);

    mat didq_dw, didq_dw_matrix_sum;

    e_hist_err = rowvec(trajectory_length, arma::fill::zeros);
    if (flag)
    {
        didq_dw = mat(2, num_weights, arma::fill::zeros);
        j_matrix = mat(trajectory_length+1, num_weights, arma::fill::zeros);
        didq_dw_matrix_sum = mat(2, num_weights, arma::fill::zeros);
    }

    mat err_integral, dudq_dw;
    colvec idq_ref, idq_refi;

    // outputs of net_action
    mat o3, udq, ndq, dnet_dw, dnet_didq, dnet_dhist_err;

    for (int len = 0; len < trajectory_length; len++)
    {
        err_integral = ts * (arma::sum(hist_err, 1) - hist_err.col(len) / 2.0);
        idq_ref = idq_ref_total.rows((trajectory_number - 1) * 2, (trajectory_number) * 2 - 1).col(len);
		hist_err.col(len) = (idq - idq_ref); // when the error is too small, the calculation becomes inaccurate
		e_hist_err.col(len) = hist_err.col(len).at(0, 0); // the calculation process of pow is slightly from that in Matlab
		//e_hist_err.col(len) = pow((arma::sum(hist_err.col(len) % hist_err.col(len))), cost_term_power / 2.0);// the calculation process of pow is slightly from that in Matlab
		net_action(idq, idq_ref, err_integral, w1, w2, w3, false, use_shortcuts, use_idq, gain1, gain2, gain3, ndq, dnet_dw, dnet_didq, dnet_dhist_err);
		udq = ndq * vmax - vdq;

        if (flag)
        {
            net_action(idq, idq_ref, err_integral, w1, w2, w3, flag, use_shortcuts, use_idq, gain1, gain2, gain3, o3, dnet_dw, dnet_didq, dnet_dhist_err);
            dudq_dw = vmax * (dnet_dw + dnet_didq * didq_dw + (ts * dnet_dhist_err) * (didq_dw_matrix_sum));
            didq_dw = a * didq_dw + b * dudq_dw;
            didq_dw_matrix_sum = didq_dw_matrix_sum + didq_dw;
        }

        idq = a * idq + b * udq;
        if (flag)
        {
            idq_refi = idq_ref_total.rows((trajectory_number - 1) * 2, (trajectory_number) * 2 - 1).col(len + 1);
            j_matrix.row(len+1) = (idq-idq_refi).t() * didq_dw * cost_term_power * pow(arma::sum((idq - idq_refi) % (idq - idq_refi)), cost_term_power / 2.0 - 1);
        }

    }

    j_total = arma::sum(e_hist_err % e_hist_err);
    if (flag)
    {
        j_matrix.shed_row(j_matrix.n_rows-1);
    }
}

// ============================================================================
// Implementation of much of the logic in LMBP.m
void train_weights_using_lm(
    mat &w1,
    mat &w2,
    mat &w3,
    double mu,
    const bool use_shortcuts,
    const bool use_idq,
    const double &num_samples,
    const double &trajectory_length,
    const double &num_iterations,
    const colvec &vdq,
    const double &mu_max,
    const double &mu_inc,
    const double &mu_dec,
    const double &mu_min,
    const double &min_grad,
    const double &ts,
    const double &imax,
    const double &iq_max,
    const double &vmax,
    const double &xl,
    const double gain1,
    const double gain2,
    const double gain3,
    const double cost_term_power,
    mat a = mat(),
    mat b = mat(),
    mat c = mat(),
    mat d = mat(),
    mat idq_start_positions = mat(),
    mat idq_ref_total = mat()
)
{
    // FIXME: Normally, a, b, c, and d would be computed by this logic instead of being parameters
    // A=-[R/L -ws; ws R/L]; B=-1/L*[1 0; 0 1]; C=[1 0; 0 1]; D=[0 0; 0 0];
    // %sysC=ss(A, B, C, D);                    % Continuous state space model
    // %sysD=c2d(sysC, Ts, 'zoh');              % Discrete state space model
    // %[a,b,c,d,Ts]=ssdata(sysD);               % Get and display matrix of discrete state space model

    double num_weights = w1.size() + w2.size() + w3.size();

    colvec idq, dw, dw_y, rr;
    mat w1_temp, w2_temp, w3_temp;
    mat j_matrix, j_matrix2;
    rowvec e_hist_err, e_hist_err2;
    mat jj, ii, h_matrix;
    double j_total, j_total_sum, j_total2, j_total_sum2;
    bool p1, p2;

    mat L = mat(num_weights, num_weights, arma::fill::zeros);
    mat R = mat(num_weights, num_weights, arma::fill::zeros);
    rowvec hist_err_total = rowvec(trajectory_length * num_samples, arma::fill::zeros);
    mat j_matrix_total = mat(trajectory_length * num_samples, num_weights, arma::fill::zeros);

    // NOTE: For purposes of testing, we want to be able to supply these as arguments
    // to fix the randomness in calculate_idq_ref
    if (idq_ref_total.empty())
    {
        idq_ref_total = mat(2 * num_samples, trajectory_length+1, arma::fill::zeros);
        for (int i = 0; i < num_samples; i++)
        {
            for (int time_step = 0; time_step < trajectory_length + 1; time_step++)
            {
                idq_ref_total.rows(i * 2, (i + 1) * 2 - 1).col(time_step) = calculate_idq_ref(
                    i, time_step, ts, imax, iq_max //, vdq, vmax, xl
                );
            }
        }
    }

    if (idq_start_positions.empty())
    {
        idq_start_positions = 0.1 * mat(2, num_samples, arma::fill::randn);
    }

    for (int iteration = 1; iteration < num_iterations + 1; iteration++)
    {
        while (mu < mu_max)
        {
            j_total_sum = 0;
            hist_err_total.zeros();
            j_matrix_total.zeros();

            for (int i = 0; i < num_samples; i++)
            {
                idq = idq_start_positions.col(i);

                unroll_trajectory_full(
                    idq, vdq, i + 1, trajectory_length,
                    w3, w2, w1,
                    true, use_shortcuts, use_idq,
                    idq_ref_total,
                    a, b,
                    vmax, ts,
                    gain1, gain2, gain3,
                    cost_term_power,
                    j_total, e_hist_err, j_matrix
                );

                j_total_sum += j_total;
                hist_err_total.cols(i * (trajectory_length - 1), (i+1) * (trajectory_length - 1) - 1) = e_hist_err.cols(1, e_hist_err.n_cols - 1);
                j_matrix_total.rows(i * (trajectory_length - 1), (i+1) * (trajectory_length - 1) - 1) = j_matrix.rows(1, j_matrix.n_rows - 1);
            }

            jj = j_matrix_total.t() * j_matrix_total;
            ii = -1 * j_matrix_total.t() * arma::vectorise(hist_err_total);
            h_matrix = jj + mu * arma::eye(num_weights, num_weights);

            // cholensky decomposition to solve for weight updates
            p1 = arma::chol(L, h_matrix, "lower");
            p2 = arma::chol(R, h_matrix, "upper");

            while (!p1 || !p2)
            {
                mu = mu * mu_inc;
				// FIXME: Should be greater than or equal?
				// What happens when mu_max is not some multiple of mu_inc?
                if (mu >= mu_max)
                {
                    break;
                }
                h_matrix = jj + mu * arma::eye(num_weights, num_weights);
                p1 = arma::chol(L, h_matrix, "lower");
                p2 = arma::chol(R, h_matrix, "upper");
            }

            if (mu >= mu_max)
            {
                cout << "reach mu_max1" << endl;
                break;
            }

            arma::solve(dw_y, L, ii) && arma::solve(dw, R, dw_y);

            w3_temp=w3+arma::reshape(dw.rows(0, w3.n_elem), w3.n_cols, w3.n_rows).t();
            w2_temp=w2+arma::reshape(dw.rows(w3.n_elem, w3.n_elem + w2.n_elem), w2.n_cols, w2.n_rows).t();
            w1_temp=w1+arma::reshape(dw.rows(w3.n_elem + w2.n_elem, dw.n_elem - 1), w1.n_cols, w1.n_rows).t();

            j_total_sum2 = 0;
            for (int i = 0; i < num_samples; i++)
            {
                idq = idq_start_positions.col(i);
                unroll_trajectory_full(
                    idq, vdq, i+1, trajectory_length,
                    w3_temp, w2_temp, w1_temp,
                    false, use_shortcuts, use_idq,
                    idq_ref_total,
                    a, b,
                    vmax, ts,
                    gain1, gain2, gain3,
                    cost_term_power,
                    j_total2, e_hist_err2, j_matrix2
                );
                j_total_sum2 += j_total2;
            }

            if (j_total_sum2 < j_total_sum)
            {
                w3 = w3_temp;
                w2 = w2_temp;
                w1 = w1_temp;
                rr = join_cols(rr, colvec(j_total_sum2 / trajectory_length / num_samples));
                mu = std::max(mu * mu_dec, mu_min);
                cout << setprecision(16) << setfill('0')
                     << "iteration: " << iteration
                     << ", mu=" << mu
                     << ", J_total_sum =" << j_total_sum / trajectory_length / num_samples
                     << ", J_total_sum2=" << j_total_sum2 / trajectory_length / num_samples << endl;
                break;
            }
            mu = mu * mu_inc;
        }

        double current_gradient = arma::min(arma::min(arma::abs(2 * j_matrix_total.t() * arma::vectorise(hist_err_total))));
        if (current_gradient < min_grad)
        {
            cout << "reach min_gra " << endl;
            break;
        }

        if (mu >= mu_max)
        {
            cout << "reach mu_max " << endl;
            break;
        }
    }
}

#endif