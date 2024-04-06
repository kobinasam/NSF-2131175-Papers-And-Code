
function run_test(jj,iteration,costTermPower)

global W1 W2 W3 Vmax Vdq a b Gain Gain2 Gain3 Ts idq_ref_centre Imax Iq_max XL useShortcuts use_idq ;
flag=0;
useShortcuts=0;

% Data initialization

Vd=690; 
delta_Vdq=[-0.3*Vd; -Vd*0.1];      % variation associated with q-axis voltage

L=0.002*1; R=0.012; Vdq=[Vd; 0]+delta_Vdq; fs=60; ws=2*pi*fs; XL=ws*L; 

t_final=4; Ts=0.001;                  % set initial time and sampling time

Vdc=1200; idq_ref=[10; 0]; idq_ref_centre=[0;0];

Vmax=Vdc*sqrt(3/2)/2;                   % maximum allowable voltage
Imax=500; Iq_max=(Vmax-Vd)/XL;

Gain=1000;
Gain2=100*2;
Gain3=100*2;
use_idq=0;

% PSC State-space model
% ---------------------

A=-[R/L -ws; ws R/L]; B=-1/L*[1 0; 0 1]; C=[1 0; 0 1]; D=[0 0; 0 0];
sysC=ss(A, B, C, D);                    % Continuous state space model
sysD=c2d(sysC, Ts, 'zoh');              % Discrete state space model
[a,b,~,~,Ts]=ssdata(sysD);               % Get and display matrix of discrete state space model
% 
% a=[ 0.920525055277549   0.364461652184452;-0.364461652184452   0.920525055277549];
% b=[ -0.038866915258523  -0.007401576588668;0.007401576588668  -0.038866915258523];

idq=[0;0]; 

timeStep=t_final*1.5/Ts;
hist_err=zeros(2,timeStep);
idq_ref_his=zeros(2,timeStep);
idq_his=zeros(2,timeStep);
% Test the performance of the action network with sampling time period of Ts

for i=1:timeStep
    if flag==1
       idq_ref=calculateIdq_ref(1, i);  
    else
        if i==floor(timeStep/3)
            idq_ref=[-50;20];    %%% idq_ref changes to [150;30] at t==t_final/2 and remains at this value until t==t_final.\
        elseif i==floor(2*timeStep/3)
            idq_ref=[-80;-20];    %%% idq_ref changes to [80;-20] at t==t_final and remains at this value until the end.
        end
    end
    idq_ref_his(:,i)=idq_ref;
    idq_his(:,i)=idq;   
    err_integral=Ts*(sum(hist_err,2)-hist_err(:,end)/2);
    hist_err(:,i)=idq-idq_ref;
    udq=net_action(idq,idq_ref,err_integral,W3,W2,W1,0,useShortcuts, use_idq)*Vmax+delta_Vdq-Vdq;       % runs the action network
    idq=a*idq+b*udq;                    % calculates next state from previous state and current action.
end

x=0:Ts:t_final*1.5-Ts;
plot(x,idq_his(1,:),x,idq_his(2,:),x,idq_ref_his(1,:),x,idq_ref_his(2,:));
grid
%
saveas(gcf, strcat('w_',num2str(jj),'_',num2str(iteration),'_lm','_',num2str(costTermPower*2)), 'png');
close all
%

