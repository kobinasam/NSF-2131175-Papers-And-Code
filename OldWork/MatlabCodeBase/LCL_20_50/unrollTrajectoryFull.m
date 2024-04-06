
function [J_total,e_hist_err,J_matix,idq_his,idq_ref_his]  = unrollTrajectoryFull(idq,trajectoryNumber,trajectoryLength,W3,W2,W1)

global Vmax Vdq a b Ts numWeights costTermPower use_idq ;
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
    err_integral=Ts*(sum(hist_err,2)-hist_err(:,i)/2);
    idq_ref=calculateIdq_ref(trajectoryNumber,i);idq_ref_his(:,i)=idq_ref;
    hist_err(:,i)=idq(1:2)-idq_ref;
    e_hist_err(:,i)=(sum((idq(1:2)-idq_ref).^2))^(costTermPower/2);
    udq=net_action(idq,idq_ref,err_integral,W3,W2,W1,0)*Vmax;
    %
    [~,Dnet_Dw,Dnet_Didq,Dnet_Dhist_err]=net_action(idq,idq_ref,err_integral,W3,W2,W1,1);
    if use_idq==1
        Dudq_Dw=Vmax*(Dnet_Dw+Dnet_Didq*Didq_Dw+Dnet_Dhist_err*Ts*(Didq_Dw_matrix_sum));
    else
        Dudq_Dw=Vmax*(Dnet_Dw+Dnet_Didq*Didq_Dw(1:2,:)+Dnet_Dhist_err*Ts*(Didq_Dw_matrix_sum-Didq_Dw(1:2,:)/2));
    end
    % %
    Didq_Dw=a*Didq_Dw+b*[Dvdq_Dw;Dudq_Dw;Dvdq_Dw];
    %
    Didq_Dw_matrix_sum=Didq_Dw_matrix_sum+Didq_Dw(1:2,:);
    %
    idq=a*idq+b*[Vdq;udq;0;0];
    idq_his(:,i)=idq;
    idq_refi=calculateIdq_ref(trajectoryNumber,i+1);
    J_matix(i+1,:)=(idq(1:2)-idq_refi)'*Didq_Dw(1:2,:)*(costTermPower)*(sum((idq(1:2)-idq_refi).^2))^(costTermPower/2-1);
end
J_total=sum(e_hist_err.^2);
J_matix(end,:)=[];


