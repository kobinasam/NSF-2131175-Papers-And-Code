function generate_net_action_test(flag, useShortcuts, use_idq)

    % Globals that we will grab from the matlab context
    %Gain=1000;
    %Gain2=100*2;
    %Gain3=100*2;

    filedir = "./testfiles/test_net_action/";

    idq = [0;0];
    idq_ref = [1;0];
    hist_err = [0.5; 0.5];

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
    basedir = "./testfiles/test_net_action/";
    subdir = generate_subdir(condition_labels, conditions);
    mkdir(strcat(basedir, "/", subdir, "/"));

    writematrix(W1, strcat(filedir, subdir, "/w1.csv"));
    writematrix(W2, strcat(filedir, subdir, "/w2.csv"));
    writematrix(W3, strcat(filedir, subdir, "/w3.csv"));
    
    if flag == 1
        [o3, Dnet_Dw, DnetDidq, Dnet_Dhist_err] = net_action(idq, idq_ref, hist_err, W3, W2, W1, flag, useShortcuts, use_idq);
        writematrix(o3, strcat(filedir, subdir, "/o3.csv"));
        writematrix(Dnet_Dw, strcat(filedir, subdir, "/dnet_dw.csv"));
        writematrix(DnetDidq, strcat(filedir, subdir, "/dnetdidq.csv"));
        writematrix(Dnet_Dhist_err, strcat(filedir, subdir, "/dnet_dhist_err.csv"));
    else
        o3 = net_action(idq, idq_ref, hist_err, W3, W2, W1, flag, useShortcuts, use_idq);
        writematrix(o3, strcat(filedir, subdir, "/o3.csv"));
    end
end
