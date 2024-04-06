
#include "calculateidq_ref.h"
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

    MexBase::validateInputDims("trajectoryNumber", inputs, 1, 1);
    MexBase::validateInputDims("timeStep", inputs, 1, 1);

    MexBase::validateDims("Ts", _matlabPtr->getVariable(u"Ts"), 1, 1);
    MexBase::validateDims("Imax", _matlabPtr->getVariable(u"Imax"), 1, 1);
    MexBase::validateDims("Iq_max", _matlabPtr->getVariable(u"Iq_max"), 1, 1);
    MexBase::validateDims("Vdq", _matlabPtr->getVariable(u"Vdq"), 2, 1);
    MexBase::validateDims("Vmax", _matlabPtr->getVariable(u"Vmax"), 1, 1);
    MexBase::validateDims("XL", _matlabPtr->getVariable(u"XL"), 1, 1);
}

// =========================================================================
void MexFunction::computeOutput(matlab::mex::ArgumentList inputs,
                                matlab::mex::ArgumentList outputs)
{
    // timestep is the timestep of the trajectory (integer)
    // trajectoryNumber is the number of this trajectory (this will provide a random number seed).
    double trajectory_number = inputs[_inArgIndices["trajectoryNumber"]][0];
    double time_step = inputs[_inArgIndices["timeStep"]][0];

    double ts = _matlabPtr->getVariable(u"Ts")[0];
    double imax = _matlabPtr->getVariable(u"Imax")[0];
    double iq_max = _matlabPtr->getVariable(u"Iq_max")[0];

    Array vdq_arr = _matlabPtr->getVariable(u"Vdq");
    vector<double> vdq = {vdq_arr[0], vdq_arr[1]};
    double vmax = _matlabPtr->getVariable(u"Vmax")[0];
    double xl = _matlabPtr->getVariable(u"XL")[0];

    std::vector<double> idq_ref = calculate_idq(trajectory_number, time_step, ts, imax, iq_max, vdq, vmax, xl);
    TypedArray<double> idq_ref_output = _factory.createArray<double>({1, 2});
    idq_ref_output[0] = idq_ref[0];
    idq_ref_output[1] = idq_ref[1];
    outputs[_outArgIndices["idq_ref"]] = std::move(idq_ref_output);
}
