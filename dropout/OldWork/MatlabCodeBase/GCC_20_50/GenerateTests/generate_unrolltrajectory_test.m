function generate_unrolltrajectory_test(flag, useShortcuts, use_idq)

    global Vmax Vdq a b Ts numWeights costTermPower idq_ref_total;

    % globals needed to call unrollTrajectoryFull
    Vdc=50;
    Vmax=Vdc*sqrt(3/2)/2; % Imax=3; Iq_max=(Vmax-Vd)/XL;
    Vd=20; Vdq=[Vd; 0];
    a = [0.920525055277549, 0.364461652184452; -0.364461652184452   0.920525055277549];
    b = [-0.038866915258523, -0.007401576588668; 0.007401576588668  -0.038866915258523];
    t_final=1; Ts=0.001;
    costTermPower=1/2;
    Gain=1000;
    Gain2=100*2;
    Gain3=100*2;

    % arguments needed to call unrollTrajectoryFull
    numSamples = 1;
    idq = [1.0753; 3.6678];
    trajectoryNumber = 1;
    trajectoryLength = t_final / Ts;

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

    condition_labels = ["shortcuts", "use_idq", "flag"];
    conditions = [useShortcuts, use_idq, flag];
    basedir = "./testfiles/test_unrolltrajectory/";
    subdir = generate_subdir(condition_labels, conditions);
    filepath = strcat(basedir, "/", subdir, "/");
    mkdir(filepath)

    writematrix(W1, strcat(filepath, "/w1.csv"));
    writematrix(W2, strcat(filepath, "/w2.csv"));
    writematrix(W3, strcat(filepath, "/w3.csv"));

    % more globals
    idq_ref_total=zeros(2*numSamples,trajectoryLength+1);
    numWeights = numel(W1) + numel(W2) + numel(W3);

    if flag == 1
        [J_total, e_hist_err, J_matix] = unrollTrajectoryFull(idq,trajectoryNumber,trajectoryLength,W3,W2,W1,flag,useShortcuts, use_idq);
        writematrix(J_matix, strcat(filepath, "/j_matrix.csv"));
    else
        [J_total, e_hist_err] = unrollTrajectoryFull(idq,trajectoryNumber,trajectoryLength,W3,W2,W1,flag,useShortcuts, use_idq);
    end

    writematrix(J_total, strcat(filepath, "/j_total.csv"));
    writematrix(e_hist_err, strcat(filepath, "/e_hist_err.csv"));
end
