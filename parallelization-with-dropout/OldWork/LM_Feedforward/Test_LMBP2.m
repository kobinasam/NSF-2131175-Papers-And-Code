%
clc;
clear all;
close all;
%


global W1 W2 W3 Vmax XL Vdq idq_ref Imax Iq_max a b Gain Gain2 Gain3 Ts;
global costTermPower idq_ref_centre useShortcuts numHids1 numHids2 numInputs integral_decay_constant;

Gain=20;
Gain2=100;
Gain3=100;
costTermPower=1;
integral_decay_constant=1;
idq_ref_centre=[0;0];

useShortcuts=0;
numHids1=3;
numOutputs=1;
numInputs=2;
W1=0.1*rand(numHids1,numInputs+1);
W3=0.1*rand(numOutputs,numHids1+1);

%
global numWeights

numIterations=10000;

[m1,n1]=size(W1);
[m3,n3]=size(W3);
numWeights=m1*n1+m3*n3;
R_avg=zeros(numIterations,1);
R_avg_validation=zeros(numIterations,1);

format long;
mu=0.001;mu_dec=0.1;mu_inc=10;mu_max=1e20;mu_min=1e-20;min_grad=1e-12;

NN=4;
p=[ 1 2 3 4;5 6 7 8];
t=sum(p);
totalJ_matix=[];DWW=zeros(13,1);
niteration2=0;R_average2=0;
R_average=0;V=zeros(1,NN);RR=[];
for iteration=1:1:numIterations
    while (mu <= mu_max) 
        totalJ_matix=[];DWW=zeros(13,1);
        R_average=0;
        for i=1:NN
            at0=Gain*net_action(p(:,i),W1,W3);
            [Dnet_Dw,~]=net_action_backpropagate(p(:,i),W1,W3);
            totalJ_matix=[totalJ_matix;Gain*Dnet_Dw];
            V(i)=at0-t(i);

            R_average=R_average+sum((at0-t(i)).^2);
        end
        if min(min(abs(2*totalJ_matix'*V'))) < min_grad
            fprintf('reach min_gra \n');
            break
        end
%         W3t=W3';W1t=W1'; WW=[W3t(:);W1t(:)]; 
%         WW=WW-(totalJ_matix'*totalJ_matix+mu*eye(numWeights))\(totalJ_matix'*V'); 
        WW=-(totalJ_matix'*totalJ_matix+mu*eye(numWeights))\(totalJ_matix'*V');
        mu
        totalJ_matix
        V
        WW
        break
        break
        W3_temp=W3+reshape(WW(1:m3*n3),m3,n3);
        W1_temp=W1+reshape(WW(m3*n3+1:end),m1,n1);
        
            R_average2=0;
        for i=1:NN
            at02=Gain*net_action(p(:,i),W1_temp,W3_temp);
            R_average2=R_average2+sum((at02-t(i)).^2);
        end
        
        if R_average2 < R_average
            W3=W3_temp;W1=W1_temp;
            mu=max(mu*mu_dec,mu_min);
            RR=[RR;R_average2];
            niteration2=niteration2+1;
            fprintf('\niteration2: %d,mu=%d, R_average2=%d\n',niteration2,mu,R_average2);
            break 
        else
            mu=mu*mu_inc;
        end     
    end
    if  mu > mu_max
        fprintf('reach mu_max \n');
        break
    end
end
 loglog(RR)


