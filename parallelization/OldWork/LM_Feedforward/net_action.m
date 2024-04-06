
% =========================================================== %
%               Backpropagation Through Time                  %
%              for Vector Control Application                 %
%                     NN output function                      %
%                       (October 2011)                        %
% =========================================================== %


function out = net_action(idq,W1,W3)

global Gain

output0A=(idq)/Gain;
input1=[output0A; -1];

sum1=W1*input1; o1=tanh(sum1);     % first hidden layer outputs
input3=[o1; -1];

sum3=W3*input3; out=tanh(sum3);         % network outputs