%
% This file is needed by "unrollTrajectoryFull.m" to calculate the derivatives over the NN controller.
%
function [o3,Dnet_Dw,Dnet_Didq,Dnet_Dhist_err]= net_action(idq, idq_ref, hist_err,W3,W2,W1,flag,Gain,Gain2,Gain3,useShortcuts,use_idq)
%
% function outputs
% o3:              the nn outputs
% Dnet_Dw:         the derivatives of the NN outputs over the NN weights
% Dnet_Didq:       the derivatives of the NN outputs over the dq currents
% Dnet_Dhist_err:  the derivatives of the NN outputs over the dq error integral
%
% function inputs
% idq:             the dq currents
% idq_ref:         the dq reference currents
% hist_err:        the dq error integral
% W3:              the NN weights from the second hidden layer to the output layer
% W2:              the NN weights from the first hidden layer to the second hidden layer
% W1:              the NN weights from the input layer to the first hidden layer
%
% flag:            flag =1: the file is to calculate the NN outputs and the derivatives over
%                  the NN controller; flag ~=1: the file only calculate the NN outputs.
% useShortcuts:    useShortcuts=1: the NN controller contains the shortcut connection.
% use_idq:         use_idq=1: the NN controller input layer contains the dq currents.

% global  Gain Gain2 Gain3   useShortcuts use_idq;

% the input layer
input0A=(idq)/Gain;
output0A=tanh(input0A);
input0B=(idq(1:2)-idq_ref)/Gain2; %% Note: error term = idq(1:2)-idq_ref
output0B=tanh(input0B);
input0C=hist_err/Gain3;
output0C=tanh(input0C);

% the first hidden layer
if use_idq==1;
    input1=[output0A; output0B; output0C; -1]; % input1=[output0A; output0B; -1];
else
    input1=[output0B; output0C; -1];
end
sum1=W1*input1;
o1=tanh(sum1);

% the second hidden layer
if useShortcuts==1
    input2=[o1; input1];
else
    input2=[o1; -1];
end

sum2=W2*input2;
o2=tanh(sum2);

% the output layer
if useShortcuts==1
    input3=[o2; input2];
else
    input3=[o2; -1];
end

sum3=W3*input3;
o3=tanh(sum3);
%%
if flag ==1
    %% compute Dnet_Dw
    % third layer
    Do3_Dw3=(1-o3.^2)*input3';
    Dnet_Dw=exdiag(Do3_Dw3);
    % second layer
    Do3_Do2=diag(1-o3.^2)*W3(:,1:size(W2,1));
    Do2_Dw2=exdiag((1-o2.^2)*input2');
    Do3_Dw2=Do3_Do2*Do2_Dw2;
    Dnet_Dw=[Dnet_Dw Do3_Dw2];
    % first layer
    Do2_Do1=diag(1-o2.^2)*W2(:,1:size(W1,1));
    if useShortcuts==1
        Do3_Do1_d3=diag(1-o3.^2)*W3(:,size(W2,1)+1:size(W2,1)+size(W1,1));
        Do3_Do1=Do3_Do1_d3+Do3_Do2*Do2_Do1;
    else
        Do3_Do1=Do3_Do2*Do2_Do1;
    end
    Do1_Dw1=exdiag((1-o1.^2)*input1');
    Do3_Dw1=Do3_Do1*Do1_Dw1;
    Dnet_Dw=[Dnet_Dw Do3_Dw1];
    
    %% compute Dnet_Didq
    if use_idq==1;
        Dinput1_OA_OB_Didq=[diag((1-output0A.^2)/Gain);[diag((1-output0B.^2)/Gain2),zeros(2,4)]];
    else
        Dinput1_OA_OB_Didq=diag((1-output0B.^2)/Gain2);
    end
    Do1_Dinput1_OA_OB=diag(1-o1.^2)*W1(:,1:end-3);
    if useShortcuts==1
        Do3_Dinput1_OA_OB_d3=diag(1-o3.^2)*W3(:,size(W2,1)+size(W1,1)+1:end-3);
        Do2_Dinput1_OA_OB_d2=diag(1-o2.^2)*W2(:,size(W1,1)+1:end-3);
        Do3_Dinput1_OA_OB_d2=Do3_Do2*Do2_Dinput1_OA_OB_d2;
        Do3_Dinput1_OA_OB=Do3_Do1*Do1_Dinput1_OA_OB+Do3_Dinput1_OA_OB_d3+Do3_Dinput1_OA_OB_d2;
    else
        Do3_Dinput1_OA_OB=Do3_Do1*Do1_Dinput1_OA_OB;
    end
    Dnet_Didq=Do3_Dinput1_OA_OB*Dinput1_OA_OB_Didq;
    %% compute Dnet_Dhist_err
    Dinput1_OC_Dhist_err=diag((1-output0C.^2)/Gain3);
    Do1_Dinput1_OC=diag(1-o1.^2)*W1(:,end-2:end-1);
    
    if useShortcuts==1
        Do3_Dinput1_OC_d3=diag(1-o3.^2)*W3(:,end-2:end-1);
        Do2_Dinput1_OC_d2=diag(1-o2.^2)*W2(:,end-2:end-1);
        Do3_Dinput1_OC_d2=Do3_Do2*Do2_Dinput1_OC_d2;
        Do3_Dinput1_OC=Do3_Do1*Do1_Dinput1_OC+Do3_Dinput1_OC_d3+Do3_Dinput1_OC_d2;
    else
        Do3_Dinput1_OC=Do3_Do1*Do1_Dinput1_OC;
    end
    Dnet_Dhist_err=Do3_Dinput1_OC*Dinput1_OC_Dhist_err;
end

