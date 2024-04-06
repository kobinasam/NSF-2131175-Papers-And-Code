tic;
clear all; close all;clc ;
global W1 W2 W3 Vmax XL Vdq idq_ref Imax Iq_max a b Gain Gain2 Gain3 Ts;
global  numHids1 numHids2 numInputs  numWeights useShortcuts costTermPower use_idq idq_ref_total;

% Data initialization
Vd=690;  L=0.002*1; R=0.012; Vdq=[Vd; 0]; fs=60; ws=2*pi*fs; XL=ws*L;
t_final=1; Ts=0.001;Vdc=1200; idq_ref=[0; 0]; 
Vmax=Vdc*sqrt(3/2)/2; Imax=500; Iq_max=(Vmax-Vd)/XL;

Gain=1000;
Gain2=100*2;
Gain3=100*2;
costTermPower=1/2;
use_idq=0;
% PSC State-space model

A=-[R/L -ws; ws R/L]; B=-1/L*[1 0; 0 1]; C=[1 0; 0 1]; D=[0 0; 0 0];
sysC=ss(A, B, C, D);                    % Continuous state space model
sysD=c2d(sysC, Ts, 'zoh');              % Discrete state space model
[a,b]=ssdata(sysD);               % Get and display matrix of discrete state space model

% Action network initialization: 2*6*6*2 with tanh functions for hidden and output layers
useShortcuts=0;
numHids1=6;
numHids2=6;
numOutputs=2;

if use_idq==1
    numInputs=6;
else
    numInputs=4;
end

if useShortcuts==1
    W1=0.1*rand(numHids1,numInputs+1);                            % Weights of the first hidden layer
    W2=0.1*rand(numHids2,numHids1+numInputs+1);                            % Weights of the second hidden layer
    W3=0.1*rand(numOutputs,numHids2+numHids1+numInputs+1);                            % Weights of the output layer
else
    W1=0.1*rand(numHids1,numInputs+1);                            % Weights of the first hidden layer
    W2=0.1*rand(numHids2,numHids1+1);                            % Weights of the second hidden layer
    W3=0.1*rand(numOutputs,numHids2+1);                            % Weights of the output layer
end
%
numIterations=300;
trajectoryLength=t_final/Ts;

numSamples=20;
idq_startPositions=20*randn(2,numSamples);

idq_ref_total=zeros(2*numSamples,trajectoryLength+1);
for i=1:numSamples
    for j=1:trajectoryLength+1
        idq_ref_total((i-1)*2+1:i*2,j)=calculateIdq_ref(i,j);
    end
end

[m1,n1]=size(W1);
[m2,n2]=size(W2);
[m3,n3]=size(W3);
numWeights=m1*n1+m2*n2+m3*n3;

previousDeltaZ1=zeros(size(W1)); % used by RPROP
previousDeltaZ2=zeros(size(W2)); % used by RPROP
previousDeltaZ3=zeros(size(W3)); % used by RPROP
    RR=[];
for iteration=1:1:numIterations    
    
     J_total_sum=0;
        hist_err_total=zeros(1,trajectoryLength*numSamples);
        J_matix_total=zeros(trajectoryLength*numSamples,numWeights);
        
        for i=1:numSamples 
            
            [J_total,e_hist_err,J_matix]=unrollTrajectoryFull(idq_startPositions(:,i),i,trajectoryLength,W3,W2,W1,1,useShortcuts,use_idq);
            J_total_sum=J_total_sum+J_total;
            hist_err_total(:,(i-1)*trajectoryLength+1:i*trajectoryLength)=e_hist_err;
            J_matix_total((i-1)*trajectoryLength+1:i*trajectoryLength,:)=J_matix;
        end
       
    dW=-2*(J_matix_total'*hist_err_total(:));
    
    totalDRdZ3=reshape(dW(1:m3*n3),n3,m3)';
    totalDRdZ2=reshape(dW(m3*n3+1:m3*n3+m2*n2),n2,m2)';
    totalDRdZ1=reshape(dW(m3*n3+m2*n2+1:end),n1,m1)';
  
    fprintf('\niteration: %d, J_average=%d\n',iteration,J_total_sum/trajectoryLength/numSamples);
    RR=[RR;J_total_sum/trajectoryLength/numSamples];    
    % calculate weight update using RPROP:
    deltaZ1=accelerateGradientUsingRPROP(totalDRdZ1,previousDeltaZ1);
    deltaZ2=accelerateGradientUsingRPROP(totalDRdZ2,previousDeltaZ2);
    deltaZ3=accelerateGradientUsingRPROP(totalDRdZ3,previousDeltaZ3);
    
    W1=W1+deltaZ1;
    W2=W2+deltaZ2;
    W3=W3+deltaZ3;
    
    % update state variables used by RPROP:
    previousDeltaZ1=deltaZ1;
    previousDeltaZ2=deltaZ2;
    previousDeltaZ3=deltaZ3;
end
toc;
        save(strcat('w',num2str(iteration),'RP',num2str(floor(J_total_sum2/trajectoryLength/numSamples))), 'W1', 'W2', 'W3');
        save(strcat('R',num2str(iteration),'RP',num2str(floor(J_total_sum2/trajectoryLength/numSamples))), 'RR');
% save w150R.mat W1 W2 W3;
% save idq_startPositions.mat idq_startPositions