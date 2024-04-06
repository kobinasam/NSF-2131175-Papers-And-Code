classdef BaseActivation
    methods(Abstract, Static)
        out = forward(obj)
        out = derivative(obj)
    end
end