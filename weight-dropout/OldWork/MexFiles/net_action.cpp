
#include <stdio.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

#include "net_action.h"
#include "functions.h"

using namespace matlab::data;

// =========================================================================
void MexFunction::operator()(matlab::mex::ArgumentList outputs,
                             matlab::mex::ArgumentList inputs)
{
    checkArguments(outputs, inputs);
    computeOutput(inputs, outputs);
}

// =========================================================================
void MexFunction::printError(std::string error)
{
    _matlabPtr->feval(u"error", 0,
        std::vector<Array>({
            _factory.createScalar(error)
        })
    );
}

// =========================================================================
void MexFunction::checkArguments(
    matlab::mex::ArgumentList inputs,
    matlab::mex::ArgumentList outputs)
{
    MexBase::checkArguments(inputs, outputs);
    MexBase::validateInputDims("idq_ref", inputs, 1, 2);
    MexBase::validateInputDims("idq", inputs, 2, 1);
    MexBase::validateInputDims("idq_ref", inputs, 2, 1);
    MexBase::validateInputDims("W3", inputs, 2, 7);
    MexBase::validateInputDims("W2", inputs, 6, 7);
    MexBase::validateInputDims("W1", inputs, 6, 5);
    MexBase::validateInputDims("flag", inputs, 1, 1);
    MexBase::validateInputDims("useShortcuts", inputs, 1, 1);
    MexBase::validateInputDims("use_idq", inputs, 1, 1);
}

// =========================================================================
void MexFunction::computeOutput(matlab::mex::ArgumentList inputs,
                                matlab::mex::ArgumentList outputs)
{
    // Get the inputs from matlab
    matlab::data::TypedArray<double> idq = std::move(inputs[_inArgIndices["idq"]]);
    matlab::data::TypedArray<double> idq_ref = std::move(inputs[_inArgIndices["idq_ref"]]);
    matlab::data::TypedArray<double> hist_err = std::move(inputs[_inArgIndices["hist_err"]]);

    matlab::data::TypedArray<double> weights1 = std::move(inputs[_inArgIndices["W1"]]);
    matlab::data::TypedArray<double> weights2 = std::move(inputs[_inArgIndices["W2"]]);
    matlab::data::TypedArray<double> weights3 = std::move(inputs[_inArgIndices["W3"]]);

    // FIXME: Right now, we implement no special logic for these flags
    bool flag = std::move(inputs[_inArgIndices["flag"]])[0];
    bool use_shortcuts = std::move(inputs[_inArgIndices["useShortcuts"]])[0];
    bool use_idq = std::move(inputs[_inArgIndices["use_idq"]])[0];

    // Create the arrays in matlab for our outputs
    matlab::data::TypedArray<double> output_vals3 = _factory.createArray<double>({1, 2});
    matlab::data::TypedArray<double> Dnet_Dw = _factory.createArray<double>({2, 86});
    matlab::data::TypedArray<double> Dnet_Didq = _factory.createArray<double>({2, 2});
    matlab::data::TypedArray<double> Dnet_Dhist_err = _factory.createArray<double>({2, 2});

    // The actual business logic here
    net_action(idq, idq_ref, hist_err, weights1, weights2, weights3, output_vals3, Dnet_Dw, Dnet_Didq, Dnet_Dhist_err);

    // Assign to outputs to return these values to matlab
    outputs[_outArgIndices["o3"]] = std::move(output_vals3);
    outputs[_outArgIndices["Dnet_Dw"]] = std::move(Dnet_Dw);
    outputs[_outArgIndices["Dnet_Didq"]] = std::move(Dnet_Didq);
    outputs[_outArgIndices["Dnet_Dhist_err"]] = std::move(Dnet_Dhist_err);
}
