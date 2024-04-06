
function generate_lmbp_test(useShortcuts, use_idq)

    global W1 W2 W3 Vmax XL Vdq idq_ref_total Imax Iq_max a b Gain Gain2 Gain3 Ts;
    global    numHids1 numHids2 numInputs  numWeights costTermPower;

    condition_labels = ["shortcuts", "use_idq"];
    conditions = [useShortcuts, use_idq];
    basedir = "./testfiles/test_lmbp";
    subdir = generate_subdir(condition_labels, conditions);
    filepath = strcat(basedir, "/", subdir, "/");
    mkdir(filepath);

    % Data initialization
    Vd=20;  L=0.025*1; R=0.5/2; Vdq=[Vd; 0]; fs=60; ws=2*pi*fs; XL=ws*L;
    t_final=1; Ts=0.001;Vdc=50; idq_ref=[0; 0];
    Vmax=Vdc*sqrt(3/2)/2; Imax=3; Iq_max=(Vmax-Vd)/XL;

    Gain=1000;
    Gain2=0.5;
    Gain3=0.5;
    costTermPower=1/2;

    % PSC State-space model
    a = [0.920525055277549, 0.364461652184452; -0.364461652184452   0.920525055277549];
    b = [-0.038866915258523, -0.007401576588668; 0.007401576588668  -0.038866915258523];
    c = [1, 0; 0, 1];
    d = [0, 0; 0, 0];

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

    numIterations=2^7;
    trajectoryLength=t_final/Ts;

    numSamples=1;
    idq_startPositions=2*randn(2,numSamples);

    idq_ref_total=zeros(2*numSamples,trajectoryLength+1);
    for i=1:numSamples
        for j=1:trajectoryLength+1
            idq_ref_total((i-1)*2+1:i*2,j)=calculateIdq_ref(i,j);
        end
    end

    writematrix(idq_startPositions, strcat(filepath, 'idq_startPositions.csv'));
    writematrix(idq_ref_total, strcat(filepath, 'idq_ref_total.csv'));
    writematrix(a, strcat(filepath, "a.csv"));
    writematrix(b, strcat(filepath, "b.csv"));
    writematrix(c, strcat(filepath, "c.csv"));
    writematrix(d, strcat(filepath, "d.csv"));

    writematrix(W1, strcat(filepath, "starting_w1.csv"));
    writematrix(W2, strcat(filepath, "starting_w2.csv"));
    writematrix(W3, strcat(filepath, "starting_w3.csv"));
    [m1,n1]=size(W1);
    [m2,n2]=size(W2);
    [m3,n3]=size(W3);
    numWeights=m1*n1+m2*n2+m3*n3;

    mu=0.001;mu_dec=0.1;mu_inc=10;mu_max=1e10;min_grad=1e-6;mu_min=1e-20;
    RR=[];
    for iteration=1:numIterations
        while (mu < mu_max)

            J_total_sum=0;
            hist_err_total=zeros(1,trajectoryLength*numSamples);
            J_matix_total=zeros(trajectoryLength*numSamples,numWeights);

            for i=1:numSamples

                [J_total,e_hist_err,J_matix]=unrollTrajectoryFull(idq_startPositions(:,i),i,trajectoryLength,W3,W2,W1,1,useShortcuts, use_idq);
                J_total_sum=J_total_sum+J_total;
                hist_err_total(:,(i-1)*(trajectoryLength-1)+1:i*(trajectoryLength-1))=e_hist_err(2:end);
                J_matix_total((i-1)*(trajectoryLength-1)+1:i*(trajectoryLength-1),:)=J_matix(2:end,:);
            end

            jj=J_matix_total'*J_matix_total;
            ii=-J_matix_total'*hist_err_total(:);
            H_matirx=jj+mu*eye(numWeights);

            [L,p1] = chol(H_matirx,'lower');
            [R,p2]= chol(H_matirx,'upper');
            while p1~=0 || p2~=0
                mu=mu*mu_inc;
                if  mu == mu_max
                    break;
                end
                H_matirx=jj+mu*eye(numWeights);
                [L,p1] = chol(H_matirx,'lower');
                [R,p2]= chol(H_matirx,'upper');
            end
            if  mu == mu_max
                fprintf('reach mu_max1 \n');
                break
            end

            dW_y=L\ii;
            dW=R\dW_y;

            W3_temp=W3+reshape(dW(1:m3*n3),n3,m3)';
            W2_temp=W2+reshape(dW(m3*n3+1:m3*n3+m2*n2),n2,m2)';
            W1_temp=W1+reshape(dW(m3*n3+m2*n2+1:end),n1,m1)';

            J_total_sum2=0;
            for i=1:numSamples
                J_total2= unrollTrajectoryFull(idq_startPositions(:,i),i,trajectoryLength,W3_temp,W2_temp,W1_temp,0,useShortcuts, use_idq);
                J_total_sum2=J_total_sum2+J_total2;
            end
            if J_total_sum2 < J_total_sum
                W3=W3_temp;W2=W2_temp;W1=W1_temp;
                RR=[RR;J_total_sum2/trajectoryLength/numSamples];
                mu=max(mu*mu_dec,mu_min);
                fprintf('\niteration: %d, mu=%d, J_total_sum2=%d\n',iteration,mu,J_total_sum2/trajectoryLength/numSamples);
                break
            end
            mu=mu*mu_inc;
        end
        if min(min(abs(2*J_matix_total'*hist_err_total(:)))) < min_grad
            fprintf('reach min_gra \n');
            break
        end

        if  mu == mu_max
            fprintf('reach mu_max \n');
            break
        end
    end

    writematrix(W1, strcat(filepath, "ending_w1.csv"));
    writematrix(W2, strcat(filepath, "ending_w2.csv"));
    writematrix(W3, strcat(filepath, "ending_w3.csv"));
end

