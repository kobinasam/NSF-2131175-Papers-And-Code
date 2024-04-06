
#include "functions.h"

int main()
{
    std::ofstream diff1("diff1.txt");
    std::ofstream diff2("diff2.txt");
    std::ofstream diff3("diff3.txt");
    std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf

    std::cout.rdbuf(diff1.rdbuf());
    cout << "======================================================" << endl;
    cout << "SEEING RESULTS FOR ONE-TO-ONE REWRITE OF Test_LMBP.m" << endl;
    test_lmbp();

    std::cout.rdbuf(diff2.rdbuf());
    cout << "======================================================" << endl;
    cout << "SEEING RESULTS FOR JORDAN'S REWRITE USING SAME RSS LOGIC" << endl;
    jordan_rewrite_of_test_lmbp();

    std::cout.rdbuf(diff3.rdbuf());
    cout << "======================================================" << endl;
    cout << "SEEING RESULTS FOR JORDAN'S REWRITE USING WIKIPEDIA LOGIC" << endl;
    jordan_based_on_wikipedia();

    std::cout.rdbuf(coutbuf); //reset to standard output again
    return 0;
}
