#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include "assert.h"

#include "mex.hpp"
#include "mexAdapter.hpp"

using namespace matlab::data;

// =========================================================================
// Base class used by all MexFunction classes where common functionality can be implemented.
// FIXME: Ideally, operator() would also be defined here, but it doesn't work with
// mex compilation because it expects the top level class, MexFunction, to
// implement operator(), so for now that's duplicated in each derived class
class MexBase
{

public:
    // =====================================================================
    MexBase(std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr,
            const std::vector<std::string> inArgs,
            const std::vector<std::string> outArgs,
            const std::vector<std::string> boolArgs = {},
            const std::vector<std::string> doubleArgs = {})
    {
        _matlabPtr = matlabPtr;
        _inArgs = inArgs;
        _outArgs = outArgs;
        _doubleArgs = doubleArgs;
        _boolArgs = boolArgs;

        // For speedy and convenient access, we store a map of strings to ints for both
        // input and output arguments
        for (int i = 0; i < _inArgs.size(); i++)
        {
            _inArgIndices[inArgs[i]] = i;
        }
        for (int i = 0; i < _outArgs.size(); i++)
        {
            _outArgIndices[outArgs[i]] = i;
        }
    };

    // =========================================================================
    void printError(std::string error)
    {
        _matlabPtr->feval(u"error", 0,
            std::vector<Array>({
                _factory.createScalar(error)
            })
        );
    }

    // =========================================================================
    void validateInputDims(std::string arg, matlab::mex::ArgumentList inputs, int expRows, int expCols)
    {
        // Don't let the dev call this with an invalid name
        assert(_inArgIndices.find(arg) != _inArgIndices.end());
        Array arr = inputs[_inArgIndices[arg]];
        validateDims(arg, arr, expRows, expCols);
    }

    // =========================================================================
    void validateDims(std::string name, matlab::data::Array arr, int expRows, int expCols)
    {
        int numRows = arr.getDimensions()[0];
        int numCols = arr.getDimensions()[1];
        if (numRows != expRows || numCols != expCols)
        {
            std::stringstream error;
            error << name << " expected dimensions: " << "(" << expRows << ", " << expCols << ")";
            printError(error.str());
        }
    }

    // =========================================================================
    void checkArguments(
        matlab::mex::ArgumentList inputs,
        matlab::mex::ArgumentList outputs)
    {
        if (inputs.size() != _inArgs.size())
        {
            // FIXME: Should be it's own function since we duplicate this logic below
            std::stringstream ss;
            for_each(_inArgs.begin(), _inArgs.end() - 1, [&ss] (const std::string& s) { ss << s << ", "; });
            ss << _inArgs.back();
            printError("Incorrect number of input args. Expected (" + ss.str() + ")");
        }

        if (outputs.size() != _outArgs.size())
        {
            std::stringstream ss;
            for_each(_outArgs.begin(), _outArgs.end() - 1, [&ss] (const std::string& s) { ss << s << ", "; });
            ss << _outArgs.back();
            printError("Incorrect number of output args. Expected (" + ss.str() + ")");
        }

        for (std::vector<std::string>::iterator arg = _boolArgs.begin(); arg != _boolArgs.end(); arg++)
        {
            if (inputs[_inArgIndices[*arg]].getType() != ArrayType::LOGICAL)
            {
                printError("Input '" + *arg + "' must be logical (boolean) type");
            }
        }

        for (std::vector<std::string>::iterator arg = _doubleArgs.begin(); arg != _doubleArgs.end(); arg++)
        {
            if (inputs[_inArgIndices[*arg]].getType() != ArrayType::DOUBLE ||
                inputs[_inArgIndices[*arg]].getType() == ArrayType::COMPLEX_DOUBLE)
            {
                printError("Input '" + *arg + "' must be real-valued double type");
            }
        }
    }

private:

    std::vector<std::string> _inArgs;
    std::vector<std::string> _outArgs;
    std::vector<std::string> _boolArgs;
    std::vector<std::string> _doubleArgs;

protected:

    std::map<std::string, int> _inArgIndices;
    std::map<std::string, int> _outArgIndices;

    virtual void computeOutput(matlab::mex::ArgumentList inputs, matlab::mex::ArgumentList outputs) {};

    matlab::data::ArrayFactory _factory;
    std::shared_ptr<matlab::engine::MATLABEngine> _matlabPtr;
};