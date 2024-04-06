#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <armadillo>

#include "functions.h"

using namespace std;
using namespace arma;

int main()
{
    // FIXME: Would be better to generate permutations rather than writing this out manually
    vector<vector<bool>> condition_bools = {
        {true, true, true},
        {true, true, false},
        {true, false, true},
        {true, false, false},
        {false, true, true},
        {false, true, false},
        {false, false, true},
        {false, false, false}
    };

    for (int i = 0; i < condition_bools.size(); i++)
    {
        // variable inputs of net_action
        bool use_shortcuts = condition_bools[i][0];
        bool use_idq = condition_bools[i][1];
        bool flag = condition_bools[i][2];

        vector<pair<string, bool>> conditions = {
            {"shortcuts", use_shortcuts},
            {"use_idq", use_idq},
            {"flag", flag}
        };

        string basedir = "./testfiles/test_net_action";
        string subdir = generate_subdir(conditions);
        string filepath = basedir + "/" + subdir + "/";

        mat w1, w2, w3;
        load_matrix_from_csv(w1, filepath + "w1.csv");
        load_matrix_from_csv(w2, filepath + "w2.csv");
        load_matrix_from_csv(w3, filepath + "w3.csv");

        // fixed inputs of net_action
        colvec idq = colvec({0, 0});
        colvec idq_ref = colvec({1, 0});
        mat hist_err = colvec({0.5, 0.5});

        double gain1 = 1000;
        double gain2 = 100 * 2;
        double gain3 = 100 * 2;

        // actual outputs of net_action
        mat o3, dnet_dw, dnet_didq, dnet_dhist_err;

        // Now call net_action and compare outputs to expected outputs
        net_action(
            idq, idq_ref, hist_err,
            w1, w2, w3,
            flag, use_shortcuts, use_idq,
            gain1, gain2, gain3,
            o3, dnet_dw, dnet_didq, dnet_dhist_err
        );

        cout << "Testing " << filepath << endl;

        // expected outputs of net_action
        colvec expected_o3;
        mat expected_dnetdidq;
        mat expected_dnet_dw;
        mat expected_dnet_dhist_err;

        load_matrix_from_csv(expected_o3,             filepath + "o3.csv");
        show_digits_of_diff(o3, expected_o3, "O3");
        if (flag)
        {
            load_matrix_from_csv(expected_dnet_dw,        filepath + "dnet_dw.csv");
            load_matrix_from_csv(expected_dnetdidq,       filepath + "dnetdidq.csv");
            load_matrix_from_csv(expected_dnet_dhist_err, filepath + "dnet_dhist_err.csv");

            show_digits_of_diff(dnet_dw, expected_dnet_dw, "DNET DW");
            show_digits_of_diff(dnet_didq, expected_dnetdidq, "DNET DIDQ");
            show_digits_of_diff(dnet_dhist_err, expected_dnet_dhist_err, "DHIST ERR");
        }
        cout << endl;
    }
}