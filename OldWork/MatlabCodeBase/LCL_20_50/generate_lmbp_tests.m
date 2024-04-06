clear all; close all;clc ;
global W1 W2 W3 Vmax  Vdq  a b Gain Gain2 Gain3 Ts;
global  idq_ref_centre  numHids1 numHids2 numInputs  numWeights useShortcuts costTermPower use_idq;


basedir = "./testfiles/test_lmbp/";
rundir = strcat(basedir, "runs");

if ~exist(basedir, 'dir')
    mkdir(basedir);
end
if ~exist(rundir, 'dir')
    mkdir(rundir);
end

if exist("./testfiles/test_lmbp/matlab_output.txt", 'file')
    delete ./testfiles/test_lmbp/matlab_output.txt
end

diary ./testfiles/test_lmbp/matlab_output.txt

% PSC State-space model
% [a,b]=LCL_Model;               % Get and display matrix of discrete state space model
% a=[ 0.908207185278397   0.359584662443059   0.012317869999152   0.004876989741393   0.001535835375403   0.000608080242024
% -0.359584662443059   0.908207185278396  -0.004876989741394   0.012317869999153  -0.000608080242024   0.001535835375403
% 0.012317869999152   0.004876989741394   0.908207185278397   0.359584662443059  -0.001535835375403  -0.000608080242024
% -0.004876989741393   0.012317869999152  -0.359584662443059   0.908207185278396   0.000608080242024  -0.001535835375403
% -17.452674720487025  -6.910002750276750  17.452674720487025   6.910002750276751   0.895505356435394   0.354555652641159
% 6.910002750276750 -17.452674720487007  -6.910002750276750  17.452674720487007  -0.354555652641159   0.895505356435395];
% b=[   0.018588725418068   0.003373795750189  -0.020278189840455  -0.004027780838478  -0.000000121766387   0.000000407513704
% -0.003373795750189   0.018588725418068   0.004027780838478  -0.020278189840455  -0.000000407513704  -0.000000121766387
% 0.020278189840455   0.004027780838478  -0.018588725418068  -0.003373795750189   0.000000121766387  -0.000000407513704
% -0.004027780838478   0.020278189840455   0.003373795750189  -0.018588725418068   0.000000407513704   0.000000121766387
% 0.055348357536563  -0.185233501741601   0.055348357536563  -0.185233501741601  -0.000042206168963  -0.000016451505633
% 0.185233501741601   0.055348357536562   0.185233501741601   0.055348357536562   0.000016451505633  -0.000042206168963];

numRuns = 1;
for jjj=1:numRuns

    fprintf(strcat("Run: ", int2str(jjj), "\n"));
    current_rundir = strcat(rundir, "/", int2str(jjj), "/");
    if ~exist(current_rundir, 'dir')
        mkdir(current_rundir);
    end
    % Data initialization
    Vd=20;  Vdq=[Vd; 0];
    t_final=1; Ts=0.001;                  % set initial time and sampling time
    Vdc=50;
    Vmax=Vdc*sqrt(3/2)/2;                   % maximum allowable voltage
    % Imax=250; Iq_max=(Vmax-Vd)/XL;

    idq_ref_centre=[0;0];
    Gain=1000;
    Gain2=0.5;
    Gain3=0.5;
    costTermPower=1/2;

    % Action network initialization: 2*6*6*2 with tanh functions for hidden and output layers
    use_idq=0;
    useShortcuts=0;
    numHids1=6;
    numHids2=6;
    numOutputs=2;
    if use_idq==1
        numInputs=10;
    else
        numInputs=4;
    end

%     if useShortcuts==1
%         W1=0.1*rand(numHids1,numInputs+1);                            % Weights of the first hidden layer
%         W2=0.1*rand(numHids2,numHids1+numInputs+1);                            % Weights of the second hidden layer
%         W3=0.1*rand(numOutputs,numHids2+numHids1+numInputs+1);                            % Weights of the output layer
%     else
%         W1=0.1*rand(numHids1,numInputs+1);                            % Weights of the first hidden layer
%         W2=0.1*rand(numHids2,numHids1+1);                            % Weights of the second hidden layer
%         W3=0.1*rand(numOutputs,numHids2+1);                            % Weights of the output layer
%     end

   % save(strcat(num2str(jjj),'w_Lm_ini.mat'), 'W1', 'W2', 'W3');
    %
    numIterations=200;
    trajectoryLength=t_final/Ts;

    numSamples=10;
%     idq_startPositions=randn(6,numSamples);
    % for i=1:numSamples
    %     idq_ref=calculateIdq_ref(i,1, basedir);
    %     idq_startPositions(:,i)=idq_startPositions(:,i)+idq_ref;
    % end

%     writematrix(W1, strcat(current_rundir, "starting_w1.csv"));
%     writematrix(W2, strcat(current_rundir, "starting_w2.csv"));
%     writematrix(W3, strcat(current_rundir, "starting_w3.csv"));
%     writematrix(a, strcat(current_rundir, 'a.csv'));
%     writematrix(b, strcat(current_rundir, 'b.csv'));
%     writematrix(idq_startPositions, strcat(current_rundir, 'idq_startPositions.csv'));
    W1 = readmatrix(strcat(current_rundir, "starting_w1.csv"));
    W2 = readmatrix(strcat(current_rundir, "starting_w2.csv"));
    W3 = readmatrix(strcat(current_rundir, "starting_w3.csv"));
    a = readmatrix(strcat(current_rundir, 'a.csv'));
    b = readmatrix(strcat(current_rundir, 'b.csv'));
    idq_startPositions = readmatrix(strcat(current_rundir, 'idq_startPositions.csv'));

    [m1,n1]=size(W1);
    [m2,n2]=size(W2);
    [m3,n3]=size(W3);
    numWeights=m1*n1+m2*n2+m3*n3;

    mu=0.001;mu_dec=0.1;mu_inc=10;mu_max=1e10;min_grad=1e-5;mu_min=1e-20;
    RR=[];
    for iteration=1:numIterations
        while (mu < mu_max)

            J_total_sum=0;
            hist_err_total=zeros(1,trajectoryLength*numSamples);
            J_matix_total=zeros(trajectoryLength*numSamples,numWeights);

            for i=1:numSamples
                [J_total,e_hist_err,J_matix, idq_his, idq_ref_his]=unrollTrajectoryFull(idq_startPositions(:,i),i,trajectoryLength,W3,W2,W1, basedir);

                %J_total
                %idq_ref_his(1:2, 1:3)
                % e_hist_expr = e_hist_err(:, 1:5)
                % j_matrix_expr = J_matix(1:5, 1:5)

                %diary off;
                %return;

                J_total_sum=J_total_sum+J_total;
                hist_err_total(:,(i-1)*trajectoryLength+1:i*trajectoryLength)=e_hist_err;
                J_matix_total((i-1)*trajectoryLength+1:i*trajectoryLength,:)=J_matix;
            end
            %          dW=-(2/mu)*(J_matix_total'*hist_err_total(:));
            %             dW=-(J_matix_total'*J_matix_total+mu*eye(numWeights))\(J_matix_total'*hist_err_total(:));

            jj=J_matix_total'*J_matix_total;
            ii=-J_matix_total'*hist_err_total(:);
            H_matirx=jj+mu*eye(numWeights);

            [L,p1] = chol(H_matirx,'lower');
            [R,p2]= chol(H_matirx,'upper');
            %         p1
            %         p2
            while p1~=0 || p2~=0
                mu=mu*mu_inc;
                if  mu == mu_max
                    break;
                end
                H_matirx=jj+mu*eye(numWeights);
                [L,p1] = chol(H_matirx,'lower');
                [R,p2]= chol(H_matirx,'upper');
                %             p1
                %             p2
            end
            if  mu == mu_max
%                 fprintf('reach mu_max1 \n');
                break
            end

            % Ly=b
            dW_y=L\ii;
            %L*x=y
            dW=R\dW_y;

            W3_temp=W3+reshape(dW(1:m3*n3),n3,m3)';
            W2_temp=W2+reshape(dW(m3*n3+1:m3*n3+m2*n2),n2,m2)';
            W1_temp=W1+reshape(dW(m3*n3+m2*n2+1:end),n1,m1)';

            J_total_sum2=0;
            for i=1:numSamples
                J_total2= unrollTrajectoryFull(idq_startPositions(:,i),i,trajectoryLength,W3_temp,W2_temp,W1_temp, basedir);
                J_total_sum2=J_total_sum2+J_total2;
            end
            if J_total_sum2 < J_total_sum
                W3=W3_temp;W2=W2_temp;W1=W1_temp;
                RR=[RR;J_total_sum2/trajectoryLength/numSamples];
                mu=max(mu*mu_dec,mu_min);
                format long;
                fprintf('iteration: %d, mu=%f, J_total_sum=%.16f, J_total_sum2=%.16f\n',iteration,mu, J_total_sum / trajectoryLength / numSamples, J_total_sum2 / trajectoryLength / numSamples);
                break
            end
            mu=mu*mu_inc;
        end
        %
        if  mu == mu_max
            fprintf('reach mu_max \n');
            break
        end
    end

%     if  J_total_sum2/trajectoryLength/numSamples < 0.1
%         save(strcat('w',num2str(iteration),'Lm_',strrep(num2str((J_total_sum2/trajectoryLength/numSamples),3),'.','_')), 'W1', 'W2', 'W3');
%         save(strcat('R',num2str(iteration),'Lm_',strrep(num2str((J_total_sum2/trajectoryLength/numSamples),3),'.','_')), 'RR');
%     else
%         delete(strcat(num2str(jjj),'w_Lm_ini.mat'));
%     end
    % run_test(jj,iteration,costTermPower);
end
% save w187Lm.mat W1 W2 W3;

% save(strcat('w_',num2str(jj),'_',num2str(iteration),'_Lm','_',num2str(costTermPower*2)), 'W1', 'W2', 'W3');
% run_test(jj,iteration,costTermPower);
% close all;
% end

diary off;

