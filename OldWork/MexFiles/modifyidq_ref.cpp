#include <vector>

#include "modifyidq_ref.h"
#include "functions.h"

using namespace matlab::data;

// =====================================================================
void MexFunction::operator()(matlab::mex::ArgumentList outputs,
                             matlab::mex::ArgumentList inputs)
{
    checkArguments(inputs, outputs);
    computeOutput(inputs, outputs);
}

// =========================================================================
void MexFunction::checkArguments(
    matlab::mex::ArgumentList inputs,
    matlab::mex::ArgumentList outputs)
{
    MexBase::checkArguments(inputs, outputs);
    MexBase::validateInputDims("idq_ref", inputs, 1, 2);

    MexBase::validateDims("Vdq", _matlabPtr->getVariable(u"Vdq"), 2, 1);
    MexBase::validateDims("Vmax", _matlabPtr->getVariable(u"Vmax"), 1, 1);
    MexBase::validateDims("XL", _matlabPtr->getVariable(u"XL"), 1, 1);
}

// =========================================================================
void MexFunction::computeOutput(matlab::mex::ArgumentList inputs,
                                matlab::mex::ArgumentList outputs)
{

    TypedArray<double> idq_ref = inputs[_inArgIndices["idq_ref"]];

    Array vdq = _matlabPtr->getVariable(u"Vdq");
    double vmax = _matlabPtr->getVariable(u"Vmax")[0];
    double xl = _matlabPtr->getVariable(u"XL")[0];

    std::vector<double> modified_idq = modify_idq(
        {idq_ref[0], idq_ref[1]},
        {vdq[0], vdq[1]},
        vmax, xl);

    TypedArray<double> modified_idq_ref = _factory.createArray<double>({1, 2});
    modified_idq_ref[0] = modified_idq[0];
    modified_idq_ref[1] = modified_idq[1];
    outputs[_outArgIndices["idq_ref"]] = std::move(modified_idq_ref);
}
