
#include <stdio.h>
#include <stdlib.h>

#include "mex.hpp"
#include "mexAdapter.hpp"

#include "mexfuncbase.h"

using namespace matlab::data;

// =============================================================================
// This class is compiled using 'mex' command in a matlab terminal to produce
// an executable that can be invoked by matlab as a function.
//
// MexBase handles as much of the common functionality as possible,
// which is why we pass the input args / output arg names here to the
// base class constructor
// 
// Important methods to implement: 
// - checkArguments: use this to validate inputs and globals from matlab
// - computeOutput: use this to implement storing the output values for
//   use by matlab
// - operator() - necessary to implement the MexFunction interface
//   FIXME: Should live in MexBase but apparently not possible 
class MexFunction: public matlab::mex::Function, public MexBase
{

public:

    // Add the names of inputs and outputs to constructor
    MexFunction(): MexBase( getEngine(), {{"idq_ref"}}, {{"idq_ref"}}){}

    void printError(std::string error);
    void checkArguments(matlab::mex::ArgumentList inputs, matlab::mex::ArgumentList outputs);
    void computeOutput(matlab::mex::ArgumentList inputs, matlab::mex::ArgumentList outputs) override;
    void operator() (matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override;
};
