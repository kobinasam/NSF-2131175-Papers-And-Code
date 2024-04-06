#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

int main()
{
    int seed_num = 10000;
    srand(seed_num);
    // arma_rng::set_seed(10000);
    colvec idq_ref = mat(2, 1, arma::fill::randu);
    cout << idq_ref[0] << endl;
    cout << idq_ref[1] << endl;
}