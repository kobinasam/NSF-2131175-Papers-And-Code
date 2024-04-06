
function [J_total,e_hist_err,J_matix]  = unrollTrajectoryFull(idq,trajectoryNumber,trajectoryLength,W3,W2,W1,flag,useShortcuts, use_idq)

global Vmax Vdq a b Ts numWeights costTermPower idq_ref_total; 

hist_err=zeros(2,trajectoryLength);
if flag==1
e_hist_err=zeros(1,trajectoryLength);
Didq_Dw=zeros(2,numWeights);
J_matix=zeros(trajectoryLength+1,numWeights);
Didq_Dw_matrix_sum=zeros(2,numWeights);
end
for i=1:trajectoryLength
    err_integral=Ts*(sum(hist_err,2)-hist_err(:,i)/2);
    idq_ref=idq_ref_total((trajectoryNumber-1)*2+1:trajectoryNumber*2,i);
    hist_err(:,i)=idq-idq_ref;
    e_hist_err(:,i)=(sum((idq-idq_ref).^2))^(costTermPower/2);
    udq=net_action(idq,idq_ref,err_integral,W3,W2,W1,0,useShortcuts, use_idq)*Vmax-Vdq;
    %
    if flag==1
    
    [~,Dnet_Dw,Dnet_Didq,Dnet_Dhist_err]=net_action(idq,idq_ref,err_integral,W3,W2,W1,1,useShortcuts, use_idq);
    Dudq_Dw=Vmax*(Dnet_Dw+Dnet_Didq*Didq_Dw+Dnet_Dhist_err*Ts*(Didq_Dw_matrix_sum));
    Didq_Dw=a*Didq_Dw+b*Dudq_Dw;
    Didq_Dw_matrix_sum=Didq_Dw_matrix_sum+Didq_Dw;
    end
    %
    idq=a*idq+b*udq;
    if flag==1
    idq_refi=idq_ref_total((trajectoryNumber-1)*2+1:trajectoryNumber*2,i+1);
    J_matix(i+1,:)=(idq-idq_refi)'*Didq_Dw*(costTermPower)*(sum((idq-idq_refi).^2))^(costTermPower/2-1);
    end
end
J_total=sum(e_hist_err.^2);
if flag==1
J_matix(end,:)=[];
end

