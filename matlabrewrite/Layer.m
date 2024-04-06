classdef Layer
    properties (Access=public)
        neurons
        activation
    end
    methods
        function obj = Layer(options)
            arguments
                options.neurons (1, 1) int8
                options.activation BaseActivation
            end
            obj.neurons = options.neurons
            obj.activation = options.activation
        end
    end
end