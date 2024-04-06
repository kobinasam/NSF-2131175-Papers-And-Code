
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <assert.h>
#include <math.h>

#include <armadillo>

using namespace std;
using namespace arma;
// void check_matrix_size(const mat &M, const uword n_rows, const uword n_cols)
// {
//     assert(M.n_rows == n_rows);
//     assert(M.n_cols == n_cols);
// };

// void check_vector_length(const rowvec &V, const uword n_length)
// {
//     assert(V.n_elem == n_length);
// };

// ==================================================================
// NOTE: Utility function to print useful info about a matrix
void print_mat(const mat m, const string &name, double precision=15)
{
    cout << "============================================" << endl;
    cout << "Name: " << name << ", Rows: " << m.n_rows << ", Cols: " << m.n_cols << endl;
    for (uword i = 0; i < m.n_rows; i++)
    {
        for (uword j = 0; j < m.n_cols; j++)
        {
            cout << m.at(i, j) << setprecision(precision) << " ";
        }
        cout << endl;
    }
    if (m.is_empty())
    {
        cout << name << " is empty!";
    }
    cout << endl;
}

// ==================================================================
// NOTE: Implements net_action.m using armadillo
void net_action(
    const colvec &idq,
    const mat &w1,
    const mat &w3,
    const double gain,
    colvec &input1,
    colvec &input3,
    colvec &o1,
    colvec &out
    )
{
    input1 = (idq / gain);
    input1.insert_rows(input1.n_rows, colvec({-1}));

    o1 = w1 * input1;
    o1.transform([](double val) { return tanh(val); } );

    input3 = o1;
    input3.insert_rows(input3.n_rows, colvec({-1}));

    out = (w3 * input3);
    out.transform([](double val) { return tanh(val); } );
}

// ==================================================================
// NOTE: Implements net_action_backpropagate.m using armadillo
void net_action_backpropagate(
    const colvec &idq,
    const mat &w1,
    const mat &w3,
    const double gain,
    const double numWeights,
    const double numHids1,
    colvec &input1,
    colvec &input3,
    colvec &o1,
    colvec &out,
    mat &dnet_dw,
    mat &dnet_didq
    )
{
    // FIXME: The original net_action_backpropagate.m recalculates this stuff
    // Shouldn't be necessary unless we intend to call this without first calling net_action
    if (input1.is_empty() || input3.is_empty() || o1.is_empty() || out.is_empty())
    {
        net_action(idq, w1, w3, gain, input1, input3, o1, out);
    }

    dnet_dw = mat(out.n_rows, numWeights, arma::fill::zeros);

    // FIXME: not used for anything in original matlab script? Should be returned though?
    dnet_didq = mat(2, 2, arma::fill::zeros);

    double m1 = w1.n_rows;
    double n1 = w1.n_cols;
    double m3 = w3.n_rows;
    double n3 = w3.n_cols;

    // % second layer
    rowvec dnet_dw2;
    rowvec dnet_dw3 = (1 - out % out) * input3.t();

    dnet_dw.row(0).cols(0, n3-1) = dnet_dw3.row(0);

    mat dw3 = ((mat)(1 - out % out)).diag() * w3;

    double dnet_dw_start = n3 * m3;
    double dnet_dw_end = dnet_dw_start + n1 * m1 - 1;

    for (int i = 0; i < m3; i++)
    {
        dnet_dw2 = ((dw3.row(i).cols(0, numHids1-1).t() % colvec(1 - o1 % o1)) * input1.t()).as_row();
        dnet_dw.row(i).cols(dnet_dw_start, dnet_dw_end) = dnet_dw2;
        // print_mat(dnet_dw2, "DNET_DW2");
    }
    // print_mat(out, "OUT");
    // print_mat(dnet_dw3, "Dnet_Dw3");
    // print_mat(dw3, "DW3");
    // print_mat(dnet_dw, "DNET_DW");
}

// =====================================================================
// Used to test correctness of net_action and net_action_backpropagation
void test_net_action()
{
    double gain = 20;
    double gain2 = 100;
    double gain3 = 100;
    double costTermPower = 1;
    double integral_decay_constant = 1;

    double numHids1=3;
    double numOutputs=1;
    double numInputs=2;
    double useShortcuts=0;

    // FIXME: Uncomment these when we want to randomize weights
    // Just here to compare to matlab results
    // mat w1 = mat(numHids1, numInputs+1).transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );
    // mat w3 = mat(numOutputs, numHids1+1).transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );

    mat w1 = mat({
        {0.10511860249074981, -0.8691958077685074, 3.9107265744512154},
        {0.8052531277712192, 0.11608271973910032, 5.081202415079452},
        {0.14211713490656408, 0.2723755030710997, 4.040974596086221}
    });

    mat w3 = mat({
        {-1.7111553944356483, -0.44719687703118854, -2.6145089128562864, -5.956955188009836}
    });

    double numWeights = w1.size() + w3.size();

    colvec idq = {1, 2};
    colvec input1, input3, o1, out;

    net_action(idq, w1, w3, gain, input1, input3, o1, out);

    mat dnet_dw;
    mat dnet_didq;

    net_action_backpropagate(idq, w1, w3, gain, numWeights, numHids1, input1, input3, o1, out, dnet_dw, dnet_didq);

    print_mat(dnet_dw, "DNET_DW");
    print_mat(dnet_didq, "DNET_DIDQ");
}

// =====================================================================
// Used to test correctness of LM algorithm
void test_lmbp()
{
    double gain = 20;
    double gain2 = 100;
    double gain3 = 100;
    double costTermPower = 1;
    double integral_decay_constant = 1;

    colvec idq_ref_centre = colvec({0, 0});

    double useShortcuts=0;
    double numHids1=3;
    double numOutputs=1;
    double numInputs=2;

    // FIXME: Uncomment these when we want to randomize weights
    // Just here to compare to matlab results
    // mat w1 = mat(numHids1, numInputs+1).transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );
    // mat w3 = mat(numOutputs, numHids1+1).transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );

    mat w1 = mat({
        {0.10511860249074981, -0.8691958077685074, 3.9107265744512154},
        {0.8052531277712192, 0.11608271973910032, 5.081202415079452},
        {0.14211713490656408, 0.2723755030710997, 4.040974596086221}
    });

    mat w3 = mat({
        {-1.7111553944356483, -0.44719687703118854, -2.6145089128562864, -5.956955188009836}
    });

    double numIterations = 10000;

    double m1 = w1.n_rows;
    double n1 = w1.n_cols;
    double m3 = w3.n_rows;
    double n3 = w3.n_cols;
    double numWeights = w1.n_elem + w3.n_elem;

    colvec r_avg = colvec(numIterations, arma::fill::zeros);
    colvec r_avg_validation = colvec(numIterations, arma::fill::zeros);

    double mu = 0.001;
    double mu_dec = 0.1;
    double mu_inc = 10;
    double mu_max = 1e20;
    double mu_min = 1e-20;
    double min_grad = 1e-12;

    double NN = 4;
    mat p = mat({
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    });

    rowvec t = arma::sum(p);

    mat totalJ_matix;
    colvec dww = colvec(13, arma::fill::zeros);

    double niteration2 = 0;
    double r_average = 0;
    double r_average2 = 0;

    rowvec v = rowvec(1, NN, arma::fill::zeros);

    colvec idq, input1, input3, o1, out, rr;
    mat dnet_dw, dnet_didq;
    double at0, at02;

    for (int i = 0; i < numIterations; i++)
    {
        while (mu <= mu_max)
        {
            totalJ_matix.clear();
            dww = colvec(13, arma::fill::zeros);
            r_average = 0;

            for (int j = 0; j < NN; j++)
            {
                idq = p.rows(0, p.n_rows-1).col(j);
                net_action(idq, w1, w3, gain, input1, input3, o1, out);
                at0 = gain * out[0];
                net_action_backpropagate(idq, w1, w3, gain, numWeights, numHids1, input1, input3, o1, out, dnet_dw, dnet_didq);
                totalJ_matix.set_size(NN, dnet_dw.n_cols);
                totalJ_matix.rows(j, j) = gain * dnet_dw;
                v.at(j) = at0 - t.at(j);
                r_average += ((at0 - t.at(j)) * (at0 - t.at(j)));
            }

            if (arma::min(arma::min(arma::abs(2 * totalJ_matix.t() * v.t()))) < min_grad)
            {
                cout << "reach min_grad" << endl;
                break;
            }

            colvec ww = -1 * arma::solve((totalJ_matix.t() * totalJ_matix + mu * arma::eye(numWeights, numWeights)), (totalJ_matix.t() * v.t()));
            mat w3_temp=w3+arma::reshape(ww.rows(0, m3*n3), m3, n3);
            mat w1_temp=w1+arma::reshape(ww.rows(m3*n3, ww.n_rows-1), m1, n1);
            r_average2 = 0;

            for (int j = 0; j < NN; j++)
            {
                idq = p.rows(0, p.n_rows-1).col(j);
                net_action(idq, w1_temp, w3_temp, gain, input1, input3, o1, out);
                at02 = gain * out[0];
                r_average2 += ((at02 - t.at(j)) * (at02 - t.at(j)));
            }

            if (r_average2 < r_average)
            {
                w3 = w3_temp;
                w1 = w1_temp;
                mu = std::max(mu * mu_dec, mu_min);
                rr.insert_rows(rr.n_rows, rowvec({r_average2}));
                niteration2 += 1;
                cout << endl << "iteration2: " << niteration2 << ", mu=" << mu << ", R_average=" << r_average << ", R_average2=" << r_average2 << endl;
                break;
            }
            else
            {
                mu = mu * mu_inc;
            }
        }
        if (mu > mu_max)
        {
            cout << "reach mu_max " << endl;
            break;
        }
    }
}

// =====================================================================
void get_rss_and_weights(
    const mat &inputs,
    const mat &outputs,
    const mat &w1,
    const mat &w3,
    const mat &jacobian,
    const mat &residuals,
    const double &gain,
    const double &mu,
    double &rss,
    mat &new_w1,
    mat &new_w3)
{
    // By solving this system of linear equations (using dampening factor mu), we are determining
    // how much to update the weights with mu
    double numWeights = w1.n_elem + w3.n_elem;
    colvec weight_updates = -1 * arma::solve(
        (jacobian.t() * jacobian + mu * arma::eye(numWeights, numWeights)),
        (jacobian.t() * residuals.t())
    );

    // Once so determined, we create a new set of weights for W1 and W3 by adding the weight updates
    new_w1 = w1 + arma::reshape(weight_updates.rows(w3.n_rows * w3.n_cols, weight_updates.n_rows-1), w1.n_rows, w1.n_cols);
    new_w3 = w3 + arma::reshape(weight_updates.rows(0, w3.n_rows * w3.n_cols), w3.n_rows, w3.n_cols);

    colvec sample, input1, input3, o1, out;
    double single_out;

    // we compute the rss so the caller can decide whether to keep the weight updates
    rss = 0;
    for (int j = 0; j < inputs.n_cols; j++)
    {
        sample = inputs.rows(0, inputs.n_rows-1).col(j);
        net_action(sample, new_w1, new_w3, gain, input1, input3, o1, out);
        single_out = gain * out[0];
        rss += ((single_out - outputs.at(j)) * (single_out - outputs.at(j)));
    }
}

// =====================================================================
// Should produce same output as test_lmbp() above
void jordan_rewrite_of_test_lmbp()
{
    double gain = 20;

    // Layer 1 weights, last column holds biases
    mat w1 = mat({
        {0.10511860249074981, -0.8691958077685074, 3.9107265744512154},
        {0.8052531277712192, 0.11608271973910032, 5.081202415079452},
        {0.14211713490656408, 0.2723755030710997, 4.040974596086221}
    });

    // Layer 3 weights, last column holds biases
    mat w3 = mat({
        {-1.7111553944356483, -0.44719687703118854, -2.6145089128562864, -5.956955188009836}
    });

    double numWeights = w1.n_elem + w3.n_elem;
    double numHids1 = 3;

    double maxIterations = 10000;
    double niteration2 = 0;
    double mu = 0.001;
    double mu_inc = 10;
    double mu_max = 1e20;
    double mu_min = 1e-20;
    double min_grad = 1e-12;

    // Test inputs
    mat inputs = mat({
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    });

    // Target outputs (i.e. the neural net should learn that the appropriate output is a sum of the inputs)
    rowvec outputs = arma::sum(inputs);

    // n represents number of samples
    double n = inputs.n_cols;

    // jacobian will hold n by m matrix of partial derivatives, where n is number of samples and m is number of parameters
    mat jacobian(inputs.n_cols, numWeights, arma::fill::zeros);

    // residuals represents differences between target y and output of the neural network
    rowvec residuals = rowvec(1, n, arma::fill::zeros);

    // We need to compare three different rss values for the LM algorithm
    double rss, rss_with_mu;

    // We compute new weights with two different mu values to decide which to keep
    mat w1_with_mu, w3_with_mu;

    colvec sample, input1, input3, o1, out, rr;
    mat partial_derivatives;

    // Unused for now
    mat dnet_didq;

    // for this simple neural net, we will just have a single output
    double single_out;

    for (int currentIteration = 0; currentIteration < maxIterations; currentIteration++)
    {
        rss = 0;
        rss_with_mu = 0;

        for (int j = 0; j < n; j++)
        {
            // Compute the neural network output for each sample
            sample = inputs.rows(0, inputs.n_rows-1).col(j);
            net_action(sample, w1, w3, gain, input1, input3, o1, out);

            // Why do we multiply our output by gain?
            single_out = gain * out[0];

            // Compute the partial derivatives for each sample
            net_action_backpropagate(sample, w1, w3, gain, numWeights, numHids1, input1, input3, o1, out, partial_derivatives, dnet_didq);

            // Then add that vector of partial derivatives to the jacobian matrix
            jacobian.rows(j, j) = gain * partial_derivatives;

            // we store the differences between our target and our nn output
            // so that we can use this to solve the system of linear equations in the modified Gauss-Newtown equation
            residuals.at(j) = single_out - outputs.at(j);

            // compute the sum of residuals squared
            rss += (residuals.at(j) * residuals.at(j));
        }

        // Now that'we computed rss for the current weights, we check against mu
        get_rss_and_weights(inputs, outputs, w1, w3, jacobian, residuals, gain, mu, rss_with_mu, w1_with_mu, w3_with_mu);

        if (rss_with_mu < rss)
        {
            w3 = w3_with_mu;
            w1 = w1_with_mu;
            mu = std::max(mu / mu_inc, mu_min);
            niteration2 += 1;
            cout << endl << "iteration2: " << niteration2 << ", mu=" << mu << ", R_average=" << rss << ", R_average2=" << rss_with_mu << endl;
        }
        else
        {
            mu = mu * mu_inc;
            if (mu > mu_max)
            {
                cout << "reach mu_max " << endl;
                return;
            }
        }
    }
}


// =====================================================================
// This is a rewrite based on wikipedia that's similar to above but with different criteria for mu / rss
void jordan_based_on_wikipedia()
{
    // Levenburg's algorithm + Marquardt's Proposal to adjust learning rate mu during training:
    //     We start with an initial learning rate (mu) and a factor (mu_inc)
    //     In each iteration, we compute RSS three times:
    //          1. RSS(B), i.e. residual sum of squares using current weights
    //          2. RSS(B+d1), i.e. residual sum of squares using current weights updated with weight updates computed using mu (i.e. d1)
    //          3. RSS(B+d2), i.e. residual sum of squares using current weights updated with weight updates computed using mu / mu_inc (i.e. d2)
    //          NOTE: in (2) and (3), we compute the weight updates by using a linear approximation of the cost
    //          together with Levenburg's modification, i.e. adding lambda * I to control how large to make weight updates
    //          If (2) and (3) are greater than (1), i.e. neither mu nor mu / mu_inc improves the cost function,
    //              then we make mu larger by multiplying by by mu_inc. We can repeat this process unless we hit some maximum mu,
    //              at which point we are done trying to improve the cost function and we stop iterating
    //          Else if (2) is smaller than (3), i.e. the current mu better reduces the cost function than a smaller mu
    //              THEN we keep mu AND update the weights according to d1
    //          Else if (3) is smaller than (2), i.e. a smaller mu better reduces the cost function, then we set mu = mu / mu_inc
    //              AND we update the weights according to d2

    double gain = 20;

    // Layer 1 weights, last column holds biases
    mat w1 = mat({
        {0.10511860249074981, -0.8691958077685074, 3.9107265744512154},
        {0.8052531277712192, 0.11608271973910032, 5.081202415079452},
        {0.14211713490656408, 0.2723755030710997, 4.040974596086221}
    });

    // Layer 3 weights, last column holds biases
    mat w3 = mat({
        {-1.7111553944356483, -0.44719687703118854, -2.6145089128562864, -5.956955188009836}
    });

    double numWeights = w1.n_elem + w3.n_elem;
    double numHids1 = 3;

    double maxIterations = 10000;
    double niteration2 = 0;
    double mu = 0.001;
    double mu_inc = 10;
    double mu_max = 1e20;
    double mu_min = 1e-20;
    double min_grad = 1e-12;

    // Test inputs
    mat inputs = mat({
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    });

    // Target outputs (i.e. the neural net should learn that the appropriate output is a sum of the inputs)
    rowvec outputs = arma::sum(inputs);

    // n represents number of samples
    double n = inputs.n_cols;

    // jacobian will hold n by m matrix of partial derivatives, where n is number of samples and m is number of parameters
    mat jacobian(inputs.n_cols, numWeights, arma::fill::zeros);

    // residuals represents differences between target y and output of the neural network
    rowvec residuals = rowvec(1, n, arma::fill::zeros);

    // We need to compare rss values for the LM algorithm
    double rss, rss_with_mu;

    // We compute new weights with two different mu values to decide which to keep
    mat w1_with_mu, w3_with_mu;

    colvec sample, input1, input3, o1, out, rr;
    mat partial_derivatives;

    // Unused for now
    mat dnet_didq;

    // for this simple neural net, we will just have a single output
    double single_out;

    for (int currentIteration = 0; currentIteration < maxIterations; currentIteration++)
    {
        rss = 0;
        rss_with_mu = 0;

        for (int j = 0; j < n; j++)
        {
            // Compute the neural network output for each sample
            sample = inputs.rows(0, inputs.n_rows-1).col(j);
            net_action(sample, w1, w3, gain, input1, input3, o1, out);

            // Why do we multiply our output by gain?
            single_out = gain * out[0];

            // Compute the partial derivatives for each sample
            net_action_backpropagate(sample, w1, w3, gain, numWeights, numHids1, input1, input3, o1, out, partial_derivatives, dnet_didq);

            // Then add that vector of partial derivatives to the jacobian matrix
            jacobian.rows(j, j) = gain * partial_derivatives;

            // we store the differences between our target and our nn output
            // so that we can use this to solve the system of linear equations in the modified Gauss-Newtown equation
            residuals.at(j) = single_out - outputs.at(j);

            // compute the sum of residuals squared
            rss += (residuals.at(j) * residuals.at(j));
        }

        // See if mu / mu_inc reduces RSS
        if (mu != mu_min)
        {
            get_rss_and_weights(inputs, outputs, w1, w3, jacobian, residuals, gain, mu / mu_inc, rss_with_mu, w1_with_mu, w3_with_mu);

            if (rss_with_mu < rss)
            {
                w1 = w1_with_mu;
                w3 = w3_with_mu;
                mu = std::max(mu / mu_inc, mu_min);

                niteration2 += 1;
                cout << endl << "iteration2: " << niteration2 << ", mu=" << mu << ", R_average=" << rss << ", R_average2=" << rss_with_mu << endl;
                continue;
            }
        }

        // If not, then just keep trying with mu
        get_rss_and_weights(inputs, outputs, w1, w3, jacobian, residuals, gain, mu, rss_with_mu, w1_with_mu, w3_with_mu);

        if (rss_with_mu < rss)
        {
            w1 = w1_with_mu;
            w3 = w3_with_mu;

            niteration2 += 1;
            cout << endl << "iteration2: " << niteration2 << ", mu=" << mu << ", R_average=" << rss << ", R_average2=" << rss_with_mu << endl;
            continue;
        }

        // If neither mu nor mu / mu_inc helped, then we try to increase mu and continue our loop unless we are at max
        mu = mu * mu_inc;
        if (mu > mu_max)
        {
            cout << "reach mu_max " << endl;
            break;
        }
    }
}

// ============================================================================
// A one-to-one rewrite of LMBP.m
void get_state_space_data(
    const mat &inputA,
    const mat &inputB,
    const mat &inputC,
    const mat &inputD,
    const double &ts,
    mat &outputA,
    mat &outputB,
    mat &outputC,
    mat &outputD
)
{
    // FIXME: How do we implement this function? For now, just assigns a to A, b to B and so on
    outputA = inputA;
    outputB = inputB;
    outputC = inputC;
    outputD = inputD;
}

// ============================================================================
// A one-to-one rewrite of LMBP.m
void lmbp(
    // const mat &w1,
    // const mat &w2,
    // const mat &w3,
    // const double& vmax,
    // const double& XL,
    // const mat &vdq,
    // const mat &idq_ref_total,
    // const mat &imax,
    // const mat &iq_max,
    // const mat &a,
    // const mat &b,
    // const double &gain1,
    // const double &gain2,
    // const double &gain3,
    const double &ts,
    // const double &num_hids1,
    // const double &num_hids2,
    // const double &num_hids3,
    // const double &num_inputs,
    // const double &num_weights,
    const bool &use_shortcuts,
    // const double &cost_term_power,
    const bool &use_idq
)
{

    // Data initialization
    double vd = 20;
    double l = 0.025 * 1;
    double r = 0.5 / 2;

    colvec vdq = colvec({vd, 0});
    double fs = 60;
    double ws = 2 * M_PI * fs;
    double xl = ws * l;

    double t_final = 1;
    double ts = 0.001;
    double vdc = 50;
    colvec idq_ref = colvec({0, 0});

    double vmax = vdc * sqrt(3/2) / 2;
    double imax = 3;
    double iq_max = (vmax - vd) / xl;

    double gain1 = 1000;
    double gain2 = 0.5;
    double gain3 = 0.5;

    double cost_term_power = 1/2;

    // PSC State-space model
    mat A = -1 * mat({{r/l, -ws}, {ws, r/l}});
    mat B = -1 / l * mat({{1, 0}, {0, 1}});
    mat C = mat({{1, 0}, {0, 1}});
    mat D = mat({{0, 0}, {0, 0}});

    // FIXME: This isn't currently implemented...
    // sysC=ss(A, B, C, D);                    % Continuous state space model
    // sysD=c2d(sysC, Ts, 'zoh');              % Discrete state space model
    // [a,b,c,d,Ts]=ssdata(sysD);               % Get and display matrix of discrete state space model
    mat a, b, c, d;
    get_state_space_data(A, B, C, D, ts, a, b, c, d);

    // Action network initialization: 2*6*6*2 with tanh functions for hidden and output layers
    double num_hids1 = 6;
    double num_hids2 = 6;
    double num_outputs = 2;
    double num_inputs = use_idq ? 6: 4;

    mat w1, w2, w3;
    double num_iterations, trajectory_length;

    std::normal_distribution<double> std_norm_distribution(0.0, 1.0);
    std::default_random_engine generator;

    // NOTE: Why so many iterations?
    for (int iteration = 0; iteration < pow(2, 30); iteration++)
    {
        // initialize the weights
        if (use_shortcuts)
        {
            w1 = 0.1 * mat(num_hids1, num_inputs + 1).transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );
            w2 = 0.1 * mat(num_hids2, num_hids1 + num_inputs + 1).transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );
            w3 = 0.1 * mat(num_outputs, num_hids2 + num_hids1 + num_inputs + 1).transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );
        }
        else
        {
            w1 = 0.1 * mat(num_hids1, num_inputs + 1).transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );
            w2 = 0.1 * mat(num_hids2, num_hids1 + 1).transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );
            w3 = 0.1 * mat(num_outputs, num_hids2 + 1).transform([](double val) { return 0.1 * rand() / (RAND_MAX); } );
        }

        num_iterations = pow(2, 7);
        trajectory_length = t_final / ts;
        num_samples = 1;
        idq_start_positions = 2 * mat(2, num_samples).transform([](double val) { return std_norm_distribution(generator) });

        mat idq_ref_total = mat(2 * num_samples, trajectory_length+1, arma::fill_zeros);

        for (int sample = 0; sample < num_samples; sample++)
        {
            for (int length = 0; length < trajectory_length+1; length++)
            {
                idq_ref_total.rows(sample * 2 + 1).cols(sample * 2, length) = calculate_idq_ref(sample, length);
            }
        }
    }

// for jjj=1:2^30

//     idq_ref_total=zeros(2*numSamples,trajectoryLength+1);
//     for i=1:numSamples
//         for j=1:trajectoryLength+1
//             idq_ref_total((i-1)*2+1:i*2,j)=calculateIdq_ref(i,j);
//         end
//     end
//     [m1,n1]=size(W1);
//     [m2,n2]=size(W2);
//     [m3,n3]=size(W3);
//     numWeights=m1*n1+m2*n2+m3*n3;

//     mu=0.001;mu_dec=0.1;mu_inc=10;mu_max=1e10;min_grad=1e-6;mu_min=1e-20;
//     RR=[];
//     for iteration=1:numIterations
//         while (mu < mu_max)

//             J_total_sum=0;
//             hist_err_total=zeros(1,trajectoryLength*numSamples);
//             J_matix_total=zeros(trajectoryLength*numSamples,numWeights);

//             for i=1:numSamples
//                 [J_total,e_hist_err,J_matix]=unrollTrajectoryFull(idq_startPositions(:,i),i,trajectoryLength,W3,W2,W1,1,useShortcuts, use_idq);
//                 J_total_sum=J_total_sum+J_total;
//                 hist_err_total(:,(i-1)*(trajectoryLength-1)+1:i*(trajectoryLength-1))=e_hist_err(2:end);
//                 J_matix_total((i-1)*(trajectoryLength-1)+1:i*(trajectoryLength-1),:)=J_matix(2:end,:);
//             end
//             % Cholesky factorization
//             %         dW=-(J_matix_total'*J_matix_total+mu*eye(numWeights))\(J_matix_total'*hist_err_total(:));
//             %                 dW=-inv(J_matix_total'*J_matix_total+mu*eye(numWeights))*(J_matix_total'*hist_err_total(:));
//             jj=J_matix_total'*J_matix_total;
//             ii=-J_matix_total'*hist_err_total(:);
//             H_matirx=jj+mu*eye(numWeights);

//             [L,p1] = chol(H_matirx,'lower');
//             [R,p2]= chol(H_matirx,'upper');
//             %         p1
//             %         p2
//             while p1~=0 || p2~=0
//                 mu=mu*mu_inc;
//                 if  mu == mu_max
//                     break;
//                 end
//                 H_matirx=jj+mu*eye(numWeights);
//                 [L,p1] = chol(H_matirx,'lower');
//                 [R,p2]= chol(H_matirx,'upper');
//                 %             p1
//                 %             p2
//             end
//             if  mu == mu_max
//                 fprintf('reach mu_max1 \n');
//                 break
//             end

//             % Ly=b
//             dW_y=L\ii;
//             %L*x=y
//             dW=R\dW_y;
//             %
//             W3_temp=W3+reshape(dW(1:m3*n3),n3,m3)';
//             W2_temp=W2+reshape(dW(m3*n3+1:m3*n3+m2*n2),n2,m2)';
//             W1_temp=W1+reshape(dW(m3*n3+m2*n2+1:end),n1,m1)';

//             J_total_sum2=0;
//             for i=1:numSamples
//                 J_total2= unrollTrajectoryFull(idq_startPositions(:,i),i,trajectoryLength,W3_temp,W2_temp,W1_temp,0,useShortcuts, use_idq);
//                 J_total_sum2=J_total_sum2+J_total2;
//             end
//             if J_total_sum2 < J_total_sum
//                 W3=W3_temp;W2=W2_temp;W1=W1_temp;
//                 RR=[RR;J_total_sum2/trajectoryLength/numSamples];
//                 mu=max(mu*mu_dec,mu_min);
//                 fprintf('\niteration: %d, mu=%d, J_total_sum2=%d\n',iteration,mu,J_total_sum2/trajectoryLength/numSamples);
//                 break
//             end
//             mu=mu*mu_inc;
//         end
//         %
//         if min(min(abs(2*J_matix_total'*hist_err_total(:)))) < min_grad
//             fprintf('reach min_gra \n');
//             break
//         end

//         if  mu == mu_max
//             fprintf('reach mu_max \n');
//             break
//         end
//     end
//     if  J_total_sum2/trajectoryLength/numSamples < 0.1
//         save(strcat('w',num2str(iteration),'Lm_',strrep(num2str((J_total_sum2/trajectoryLength/numSamples),3),'.','_')), 'W1', 'W2', 'W3');
//         save(strcat('R',num2str(iteration),'Lm_',strrep(num2str((J_total_sum2/trajectoryLength/numSamples),3),'.','_')), 'RR');
//     end
//     % run_test(jj,iteration,costTermPower);
// end

// %  loglog(RR);grid;
// % save w5320Lm.mat W1 W2 W3;




}
