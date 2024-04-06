
classdef Rnncontroller

    properties
        layers
        hiddenLayers
        gain1
        gain2
        gain3
        useIdq
        initWeights
        weights
        inputSize
        outputSize
        activations
    end

    % methods (Access=private)
    %     function out = verifyIsActivation(x)
    %         return verifyInstanceOf(x)
    % end
     
    methods

        function obj = Rnncontroller(options)

            arguments
                options.hiddenLayers (1,:) Layer                    = [Layer(neurons=6, activation=Tanh), Layer(neurons=6, activation=Tanh)]
                options.gain1        (1,1) double  {mustBePositive} = 1000
                options.gain2        (1,1) double  {mustBePositive} = 2000
                options.gain3        (1,1) double  {mustBePositive} = 3000
                options.initWeights  (1,:) cell    {isnumeric}      = {}
                options.useIdq       (1,1) logical {islogical}      = false
            end

            if options.useIdq
                layers=[
                    Layer(neurons=6, activation=Tanh),  ...
                    options.hiddenLayers,               ...
                    Layer(neurons=2, activation=Tanh)   ...
                ];
            else
                layers=[
                    Layer(neurons=4, activation=Tanh),  ...
                    options.hiddenLayers,               ...
                    Layer(neurons=2, activation=Tanh)   ...
                ];
            end

            if ~isempty(options.initWeights)
                if size(options.hiddenLayers, 2) + 1 ~= size(options.initWeights, 2)
                    throw(MException('RNN:LayerWeightMismatch', 'Number of layers incomptatible with number of weight matrices'));
                end

                for i = 1:length(options.initWeights)
                    layerWeights = options.initWeights{i};
                    if size(layerWeights) ~= [layers(i+1).neurons, layers(i).neurons + 1]
                        throw(MException('RNN:LayerWeightMismatch', sprintf('Shape of weight matrix %d incompatible with shape of layers', i)));
                    end
                end
            else
                options.initWeights = cell(size(layers, 2) - 1);
                for i=1:size(layers, 2) - 1
                    options.initWeights{i} = rand(layers(i+1).neurons, layers(i).neurons+1);
                end
            end

            obj.layers          = layers;
            obj.hiddenLayers    = options.hiddenLayers;
            obj.gain1           = options.gain1;
            obj.gain2           = options.gain2;
            obj.gain3           = options.gain3;
            obj.initWeights     = options.initWeights;
            obj.weights         = options.initWeights;
            obj.useIdq          = options.useIdq;
            obj.inputSize       = layers(1).neurons;
            obj.outputSize      = layers(end).neurons;

        end

        function [modelInputs, modelOutputs] = netAction(obj, idq, idqRef, histErr, computeDerivatives)

            arguments
                obj
                idq                 (2,1)  double
                idqRef              (2,1)  double
                histErr             (2,1)  double
                computeDerivatives  (1,1)  logical = true
            end

            % preprocess the input layer with gain values and tanh
            if obj.useIdq
                input=tanh([idq/obj.gain1; (idq(1:2)-idqRef)/obj.gain2; histErr / obj.gain3]);
            else
                input=tanh([(idq(1:2)-idqRef)/obj.gain2; histErr / obj.gain3]);
            end

            nextInput = [input; -1];  % -1 for subtracting bias

            % then do the forward pass and store the computations for backwards pass
            modelOutputs = {};
            modelInputs  = {};
            for i=1:size(obj.layers, 2) - 1
                layerOutput=obj.layers(i).activation.forward(obj.weights{i} * nextInput);
                modelInputs = [modelInputs, nextInput]
                modelOutputs = [modelOutputs, layerOutput];
                nextInput = [layerOutput; -1]; % -1 for subtracting bias
            end

            % FIXME: Still need to do all the backward pass logic. We will use the obj.activations(i).derivative(X)
            % so that the code will work regardless of the activation function

            % if computeDerivatives
            %     networkDerivatives = obj.layers(i).activation.derivative(o3) * input3;
            %     %% compute Dnet_Dw
            %     % third layer
            %     Do3_Dw3=(1-o3.^2)*input3';
            %     Dnet_Dw=exdiag(Do3_Dw3);
            %     % second layer
            %     Do3_Do2=diag(1-o3.^2)*W3(:,1:size(W2,1));
            %     Do2_Dw2=exdiag((1-o2.^2)*input2');
            %     Do3_Dw2=Do3_Do2*Do2_Dw2;
            %     Dnet_Dw=[Dnet_Dw Do3_Dw2];
            %     % first layer
            %     Do2_Do1=diag(1-o2.^2)*W2(:,1:size(W1,1));
            %     if useShortcuts==1
            %         Do3_Do1_d3=diag(1-o3.^2)*W3(:,size(W2,1)+1:size(W2,1)+size(W1,1));
            %         Do3_Do1=Do3_Do1_d3+Do3_Do2*Do2_Do1;
            %     else
            %         Do3_Do1=Do3_Do2*Do2_Do1;
            %     end
            %     Do1_Dw1=exdiag((1-o1.^2)*input1');
            %     Do3_Dw1=Do3_Do1*Do1_Dw1;
            %     Dnet_Dw=[Dnet_Dw Do3_Dw1];
                
            %     %% compute Dnet_Didq
            %     if use_idq==1;
            %         Dinput1_OA_OB_Didq=[diag((1-output0A.^2)/Gain);[diag((1-output0B.^2)/Gain2),zeros(2,4)]];
            %     else
            %         Dinput1_OA_OB_Didq=diag((1-output0B.^2)/Gain2);
            %     end
            %     Do1_Dinput1_OA_OB=diag(1-o1.^2)*W1(:,1:end-3);
            %     if useShortcuts==1
            %         Do3_Dinput1_OA_OB_d3=diag(1-o3.^2)*W3(:,size(W2,1)+size(W1,1)+1:end-3);
            %         Do2_Dinput1_OA_OB_d2=diag(1-o2.^2)*W2(:,size(W1,1)+1:end-3);
            %         Do3_Dinput1_OA_OB_d2=Do3_Do2*Do2_Dinput1_OA_OB_d2;
            %         Do3_Dinput1_OA_OB=Do3_Do1*Do1_Dinput1_OA_OB+Do3_Dinput1_OA_OB_d3+Do3_Dinput1_OA_OB_d2;
            %     else
            %         Do3_Dinput1_OA_OB=Do3_Do1*Do1_Dinput1_OA_OB;
            %     end
            %     Dnet_Didq=Do3_Dinput1_OA_OB*Dinput1_OA_OB_Didq;
            %     %% compute Dnet_Dhist_err
            %     Dinput1_OC_Dhist_err=diag((1-output0C.^2)/Gain3);
            %     Do1_Dinput1_OC=diag(1-o1.^2)*W1(:,end-2:end-1);
                
            %     if useShortcuts==1
            %         Do3_Dinput1_OC_d3=diag(1-o3.^2)*W3(:,end-2:end-1);
            %         Do2_Dinput1_OC_d2=diag(1-o2.^2)*W2(:,end-2:end-1);
            %         Do3_Dinput1_OC_d2=Do3_Do2*Do2_Dinput1_OC_d2;
            %         Do3_Dinput1_OC=Do3_Do1*Do1_Dinput1_OC+Do3_Dinput1_OC_d3+Do3_Dinput1_OC_d2;
            %     else
            %         Do3_Dinput1_OC=Do3_Do1*Do1_Dinput1_OC;
            %     end
            %     Dnet_Dhist_err=Do3_Dinput1_OC*Dinput1_OC_Dhist_err;
            % end
            % end
        end
    end
end
