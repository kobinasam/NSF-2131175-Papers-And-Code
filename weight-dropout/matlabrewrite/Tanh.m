classdef Tanh < BaseActivation
    methods(Static)
        function out = forward(x)
            out = tanh(x);
        end

        function out = derivative(x)
            out = 1 - tanh(x).*2;
        end
    end
end