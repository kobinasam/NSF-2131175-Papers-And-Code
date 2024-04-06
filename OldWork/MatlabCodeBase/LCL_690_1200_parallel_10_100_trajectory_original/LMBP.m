%
% This file is the main file of LM+FATT training for RNN controller of three phase LCL-GCC, which contains the LM parts for training.
%
% The improtant settings are marked by "%%%-----%%%".
clear all; close all;clc ;
% global W1 W2 W3 Vmax  Vdq  a b Gain Gain2 Gain3 Ts;
% global  idq_ref_centre  numHids1 numHids2 numInputs  numWeights useShortcuts costTermPower use_idq;

% set up number of workers
% matlab 2010b
% matlabpool close force local
% out=findResource;
% matlabpool(out); % 4 workers

% matlab 2014a

% for workerii=1:50  % Set up the number of workers form 1 to 50 and record the running time.
% delete(gcp('nocreate'));
% parpool(workerii);
% tic;
% for numSample_time=2:10 % Set up the multiplier for the base trajectory, 10, if  numSample_time=2, then numSamples= 2*10= 20; if  numSample_time=3, then numSamples= 3*10= 30; 
diary off;
diary on;

numSample_time=1;
for jjj=1:10

    % Data initialization
    Vd=690;  Vdq=[Vd; 0];
    t_final=1; Ts=0.001;                  % set initial time and sampling time
    Vdc=1200;
    Vmax=Vdc*sqrt(3/2)/2;                 % maximum allowable voltage

    idq_ref_centre=[0;0];
    Gain=1000;
    Gain2=100*1;
    Gain3=100*1;
    costTermPower=1/2;

    % PSC State-space model

    %[a,b]=LCL_Model;                      % Get and display matrix of discrete state space model

    a=[0.922902679404235   0.365403020170600   0.001311850123628   0.000519398207289  -0.006031602076712  -0.002388080200093
  -0.365403020170600   0.922902679404235  -0.000519398207289   0.001311850123628   0.002388080200093  -0.006031602076712
   0.001311850123628   0.000519398207289   0.922902679404235   0.365403020170600   0.006031602076712   0.002388080200093
  -0.000519398207289   0.001311850123628  -0.365403020170600   0.922902679404235  -0.002388080200093   0.006031602076712
   0.120632041534246   0.047761604001858  -0.120632041534246  -0.047761604001858   0.921566702872299   0.364874069642510
  -0.047761604001858   0.120632041534245   0.047761604001858  -0.120632041534245  -0.364874069642510   0.921566702872299];
    b=[   0.488106762997528   0.093547911260568  -0.485485431756243  -0.091984091707451  -0.000001945097416   0.000009097619657
  -0.093547911260568   0.488106762997528   0.091984091707451  -0.485485431756243  -0.000009097619657  -0.000001945097416
   0.485485431756243   0.091984091707451  -0.488106762997528  -0.093547911260568   0.000001945097416  -0.000009097619657
  -0.091984091707451   0.485485431756243   0.093547911260568  -0.488106762997528   0.000009097619657   0.000001945097416
   0.038901948324797  -0.181952393142100   0.038901948324797  -0.181952393142100   0.000002613550852   0.000001600210032
   0.181952393142100   0.038901948324797   0.181952393142100   0.038901948324797  -0.000001600210032   0.000002613550852];

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
%
%     if useShortcuts==1
%         W1=0.1*rand(numHids1,numInputs+1);                            % Weights of the first hidden layer
%         W2=0.1*rand(numHids2,numHids1+numInputs+1);                   % Weights of the second hidden layer
%         W3=0.1*rand(numOutputs,numHids2+numHids1+numInputs+1);        % Weights of the output layer
%     else
%         W1=0.1*rand(numHids1,numInputs+1);                            % Weights of the first hidden layer
%         W2=0.1*rand(numHids2,numHids1+1);                             % Weights of the second hidden layer
%         W3=0.1*rand(numOutputs,numHids2+1);                           % Weights of the output layer
%     end
    W1=[     0.081472368639318   0.027849821886705   0.095716694824295   0.079220732955955   0.067873515485777
   0.090579193707562   0.054688151920498   0.048537564872284   0.095949242639290   0.075774013057833
   0.012698681629351   0.095750683543430   0.080028046888880   0.065574069915659   0.074313246812492
   0.091337585613902   0.096488853519928   0.014188633862722   0.003571167857419   0.039222701953417
   0.063235924622541   0.015761308167755   0.042176128262628   0.084912930586878   0.065547789017756
   0.009754040499941   0.097059278176062   0.091573552518907   0.093399324775755   0.017118668781156];
    W2=[   0.012649986532930   0.031747977514944   0.055573794271939   0.055778896675488   0.025779225057201   0.040218398522248   0.087111112191539
   0.013430330431357   0.031642899914629   0.018443366775765   0.031342898993659   0.039679931863314   0.062067194719958   0.035077674488589
   0.009859409271100   0.021756330942282   0.021203084253232   0.016620356290215   0.007399476957694   0.015436980547927   0.068553570874754
   0.014202724843193   0.025104184601574   0.007734680811268   0.062249725927990   0.068409606696201   0.038134520444447   0.029414863376785
   0.016825129849153   0.089292240528598   0.091380041077957   0.098793473495250   0.040238833269616   0.016113397184936   0.053062930385689
   0.019624892225696   0.070322322455629   0.070671521769693   0.017043202305688   0.098283520139395   0.075811243132742   0.083242338628518];
    W3=[     0.002053577465818   0.065369988900825   0.016351236852753   0.079465788538875   0.044003559576025   0.075194639386745   0.006418708739190
   0.092367561262041   0.093261357204856   0.092109725589220   0.057739419670665   0.025761373671244   0.022866948210550   0.076732951077657];
    %
    numIterations=1024;    %%% LM training stop condition 3: set the training max Iterations.%%%
    trajectoryLength=t_final/Ts;

    numSamples=10*numSample_time;        %%% set the number of trajectories.%%%
   % rng(0);
   % idq_startPositions=randn(6,numSamples);
   
  
    idq_startPositions_temp=[   0.475860587278343  -0.002854960144321  -0.777698538311603   0.879677164233489  -1.314723503486884   0.064516742311104  -0.037533188819172  -1.725427789528692   0.093108760058804  -0.430206242426100
   1.412232686444504   0.919867079806395   0.566696097539305   2.038876251414042  -0.416411219699434   0.600291949185784  -1.896304493622450   0.288228089665011  -0.378157056589758  -1.627322736469780
   0.022608484309598   0.149808732632761  -1.382621159480352   0.923932448688937   1.224687824785337  -1.361514954864567  -2.127976768182674  -1.594183720266807  -1.482676111059003   0.166347492460066
  -0.047869410220206   1.404933445676977   0.244474675589888   0.266917446595828  -0.043584205546333   0.347592631960065  -1.176923330714958   0.110218849223381  -0.043818585358295   0.376265910450719
   1.701334654274959   1.034121539569710   0.808438803167691   0.641661506421779   0.582423277447969  -0.181843218459334  -0.990532220484176   0.787066676357980   0.960825211682115  -0.226950464706233
  -0.509711712767427   0.291570288770806   0.213041698417000   0.425485355625296  -1.006500074619336  -0.939534765941492  -1.173032327267405  -0.002226786313836   1.738244932613340  -1.148912289618790];
 idq_startPositions=[]; 
  for istart=1:numSample_time 
      idq_startPositions=[idq_startPositions,idq_startPositions_temp];
  end
    [m1,n1]=size(W1);
    [m2,n2]=size(W2);
    [m3,n3]=size(W3);
    numWeights=m1*n1+m2*n2+m3*n3;
    % initialize LM parameters
    mu=1;mu_dec=0.1;mu_inc=10;mu_max=1e10;min_grad=1e-5;mu_min=1e-20;
    RR=[];
    for iteration=1:numIterations
        %         while (mu < mu_max)
        % initialize LM parameters
        J_total_sum=0;
        hist_err_total=zeros(1,trajectoryLength*10);
        J_matix_total=zeros(trajectoryLength*10,numWeights);
        % Use FATT to calculate total cost of each trajectory, the error vector V,  and the jacobian matrix.
        for i=1:numSamples
            [J_total(i),e_hist_err(:,i),J_matix(:,:,i)]=unrollTrajectoryFull(idq_startPositions(:,i),i,trajectoryLength,W3,W2,W1,Gain,Gain2,Gain3,useShortcuts,use_idq,Vmax,Vdq,a,b,Ts,numWeights,costTermPower);
            %                 J_total_sum=J_total_sum+J_total;
            %                 hist_err_total(:,(i-1)*trajectoryLength+1:i*trajectoryLength)=e_hist_err;
            %                 J_matix_total((i-1)*trajectoryLength+1:i*trajectoryLength,:)=J_matix;
            return;
        end

        for ii=1:10
            %                 tic

            J_total_sum=J_total_sum+J_total(ii);
            %                 toc
            hist_err_total(:,(ii-1)*(trajectoryLength-1)+1:ii*(trajectoryLength-1))=e_hist_err(2:end,ii);
            J_matix_total((ii-1)*(trajectoryLength-1)+1:ii*(trajectoryLength-1),:)=J_matix(2:end,:,ii);
        end
        
        %  dW=-(2/mu)*(J_matix_total'*hist_err_total(:));
        %  dW=-(J_matix_total'*J_matix_total+mu*eye(numWeights))\(J_matix_total'*hist_err_total(:));
        while (mu < mu_max)
            % Cholesky factorization
            jj=J_matix_total'*J_matix_total;
            ii=-J_matix_total'*hist_err_total(:);
            % while (mu < mu_max): the while loop can alos be set to start
            % from this line.
            H_matirx=jj+mu*eye(numWeights);

            [L,p1] = chol(H_matirx,'lower');
            [R,p2]= chol(H_matirx,'upper');
            %         p1
            %         p2
            % increase mu to solve the problem of negative definite matrix,
            % which is basically from the calculation error.
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
            % reconstruct the NN weight update matrix
            W3_temp=W3+reshape(dW(1:m3*n3),n3,m3)';
            W2_temp=W2+reshape(dW(m3*n3+1:m3*n3+m2*n2),n2,m2)';
            W1_temp=W1+reshape(dW(m3*n3+m2*n2+1:end),n1,m1)';

            J_total_sum2=0;
            for i=1:numSamples
                J_total2(i)= unrollTrajectoryFull(idq_startPositions(:,i),i,trajectoryLength,W3_temp,W2_temp,W1_temp,Gain,Gain2,Gain3,useShortcuts,use_idq,Vmax,Vdq,a,b,Ts,numWeights,costTermPower);
                %                 J_total_sum2=J_total_sum2+J_total2;
            end
            for ii=1:10
                J_total_sum2=J_total_sum2+J_total2(ii);
            end

            if J_total_sum2 < J_total_sum
                W3=W3_temp;W2=W2_temp;W1=W1_temp;
                RR=[RR;J_total_sum2/trajectoryLength/numSamples];
                mu=max(mu*mu_dec,mu_min);
                fprintf('iteration: %d, mu=%d, J_total_sum2=%16.15f\n',iteration,mu,J_total_sum2/trajectoryLength/10);
                break
            end
            mu=mu*mu_inc;
        end
        %         % LM training stop condition 1: the gradient reseaches the min gradient.
        %         if min(min(abs(2*J_matix_total'*hist_err_total(:)))) < min_grad
        %             fprintf('reach min_gra \n');
        %             break
        %         end
        % LM training stop condition 2: the mu reseaches the max mu.
        if  mu == mu_max
            fprintf('reach mu_max \n');
            break
        end
    end

%     if  J_total_sum2/trajectoryLength/numSamples < 12  %%% set the goal of trainig objectives.%%%
%         save(strcat('w',num2str(iteration),'Lm_',strrep(num2str((J_total_sum2/trajectoryLength/numSamples),3),'.','_')), 'W1', 'W2', 'W3');
%         save(strcat('R',num2str(iteration),'Lm_',strrep(num2str((J_total_sum2/trajectoryLength/numSamples),3),'.','_')), 'RR');
%     end
end
% record_time=toc;
% save(strcat('run_record_time',num2str(numSamples),'_',num2str(workerii)), 'record_time');
% end
% end
