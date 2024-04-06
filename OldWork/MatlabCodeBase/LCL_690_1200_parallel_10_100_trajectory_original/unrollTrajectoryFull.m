%
% This file is needed by "LMBP.m" to calculate the derivatives through the trajectory, which is the Forward Accumulation Throuhg Time (FATT)algorithm.
%
function [J_total,e_hist_err,J_matix,idq_his,idq_ref_his]  = unrollTrajectoryFull(idq,trajectoryNumber,trajectoryLength,W3,W2,W1,Gain,Gain2,Gain3,useShortcuts,use_idq,Vmax,Vdq,a,b,Ts,numWeights,costTermPower)
%
% J_total:             the total cost
% e_hist_err:          the local cost vector
% J_matix:             the jacobian matrix


% global Vmax Vdq a b Ts numWeights costTermPower use_idq ;
% Thess are corresponding to "1" line of FATT algorithm: initialize all
% the variables.

% L=0.002/2; Lg=L;Ld=Lg; Rg=0.006/1; Rd=0.006/1;
% fs=60; ws=2*pi*fs; C=50*10^(-6)*1; C=0;
%

idq_his=zeros(6,trajectoryLength);
idq_ref_his=zeros(2,trajectoryLength);
hist_err=zeros(2,trajectoryLength);
e_hist_err=zeros(1,trajectoryLength);
% %
Didq_Dw=zeros(6,numWeights);
Dvdq_Dw=zeros(2,numWeights);
% %
J_matix=zeros(trajectoryLength+1,numWeights);
Didq_Dw_matrix_sum=zeros(2,numWeights);
for i=1:trajectoryLength
    err_integral=Ts*(sum(hist_err,2)-hist_err(:,i)/2);                                                                 % calculate the error integral: corresponding to "12" line of FATT algorithm. The reason of doing this is to use "sum" function for simplification of error integral calculation.
    idq_ref=calculateIdq_ref(trajectoryNumber,i,Ts);idq_ref_his(:,i)=idq_ref;                                             % calcualte the dq reference         
    hist_err(:,i)=idq(1:2)-idq_ref;                                                                                    % calcualte the dq error
    e_hist_err(:,i)=(sum((idq(1:2)-idq_ref).^2))^(costTermPower/2);                                                    % calcualte the local cost
     udq=net_action(idq,idq_ref,err_integral,W3,W2,W1,0,Gain,Gain2,Gain3,useShortcuts,use_idq)*Vmax;                                                          % calculate the NN control outputs: corresponding to "4" line of FATT algorithm
% 
% Req=Rd+Rg-Rd*ws^2*C*Lg-Rg*ws^2*C*Ld;
%  Leq=Lg+Ld+Rd*Rg*C-ws^2*Ld*C*Lg;
% usdq=([-Req ws*Leq;-ws*Leq -Req]*idq_ref+[Vdq(1)*(1-ws^2*Ld*C);Vdq(1)*Rd*ws*C]);
%      udq=net_action(idq,idq_ref,err_integral,W3,W2,W1,0,Gain,Gain2,Gain3,useShortcuts,use_idq)*Vmax+usdq; 
    %
    [~,Dnet_Dw,Dnet_Didq,Dnet_Dhist_err]=net_action(idq,idq_ref,err_integral,W3,W2,W1,1,Gain,Gain2,Gain3,useShortcuts,use_idq);                              % calculate the derivatives over the NN controller
    % calculate the derivatives of the controller outputs over the dq currents: corresponding to "6" line of FATT 
    if use_idq==1                                                               
        Dudq_Dw=Vmax*(Dnet_Dw+Dnet_Didq*Didq_Dw+Dnet_Dhist_err*Ts*(Didq_Dw_matrix_sum));
    else
        Dudq_Dw=Vmax*(Dnet_Dw+Dnet_Didq*Didq_Dw(1:2,:)+Dnet_Dhist_err*Ts*(Didq_Dw_matrix_sum-Didq_Dw(1:2,:)/2));
    end
    % % % accumulate the next step derivatives of the dq currents over the NN weights: corresponding to "7" line of FATT algorithm
    Didq_Dw=a*Didq_Dw+b*[Dvdq_Dw;Dudq_Dw;Dvdq_Dw];
    % % accumulate the derivatives of the error integrals over the NN weights: which is needed for dsdq/dw and corresponding to "8" and "9" line of FATT algorithm. 
    Didq_Dw_matrix_sum=Didq_Dw_matrix_sum+Didq_Dw(1:2,:);
    % % accumulate the next step system states: corresponding to "10" line of FATT algorithm
%     idq=a*(idq+[randn(4,1);0;0])+b*[Vdq;udq;0;0];
        idq=a*(idq)+b*[Vdq;udq;0;0];

    
    % add saturation to dq currents
    idq(1:4)=max(min(idq(1:4), 1000*ones(4,1)),-1000*ones(4,1));
    
    %
    idq_his(:,i)=idq;
    idq_refi=calculateIdq_ref(trajectoryNumber,i+1,Ts);                                                                   % calculate the next dq reference
    J_matix(i+1,:)=(idq(1:2)-idq_refi)'*Didq_Dw(1:2,:)*(costTermPower)*(sum((idq(1:2)-idq_refi).^2))^(costTermPower/2-1); % accumulate the (k+1)th row of the Jacobian mateix: corresponding to "15" line of FATT algorithm
end
J_total=sum(e_hist_err.^2); % accumulate the total cost: corresponding to "13" line of FATT algorithm. The reason of doing this is to use "sum" function for simplification of error integral calculation.
J_matix(end,:)=[];          % remove the last row of the Jacobian matrix.


