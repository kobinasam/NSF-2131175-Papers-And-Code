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
flag = 1;
useShortcuts = 0;
use_idq = 0;

W1 = [
    0.10511860249074981 -0.8691958077685074 3.9107265744512154 3.8296505582151905 0.0431373966662373;
    0.8052531277712192 0.11608271973910032 5.081202415079452 -1.666910901747036 1.2461853285676618;
    0.14211713490656408 0.2723755030710997 4.040974596086221 2.0482239539091487 0.14506617911530964;
    -0.3952723230078815 -1.4229869215305775 4.2551314235012185 5.561432608781822 2.512020072447385E-4;
    -0.2772247462559281 0.6999358816350526 1.6369899052747476 2.6788819705306155 -0.10953612423359192;
    0.37612754523479003 1.2857167349312448 -2.973060687194107 8.095548964772654 0.028991336931277124;
];

W2 = [
    1.440539114493213 -0.27253071839005777 0.5278868902219289 1.3712226806164334 2.25513928618451 1.3948446255239006 0.3449374254994523;
    3.3789817246548357 0.19860814862310894 3.4592707210714577 -1.9110270294298068 0.2247519084049887 -1.0022103471763135 -0.4753830547340122;
    -0.3590224302196529 1.2176551644649063 3.145578151429633 1.8631207326452712 3.708974043074285 -0.09608244193951347 1.5661350150973756;
    4.031099888420788 -2.6851875859289094 -2.7498688647349647 2.7486598886675715 2.439552173754654 5.660170953027147 0.925728746264457;
    -0.6221529426089921 0.7320648747641353 4.212370496471141 -4.0812165587839555 1.5473829764451348 -6.45653407631204 -0.817547511050558;
    1.5225000509529556 -1.0360350040097754 1.7030720130819914 0.5344327232788693 0.6307629342167956 1.0930386330505284 0.07424903099042285;
];

W3 = [
    -1.7111553944356483 -0.44719687703118854 -2.6145089128562864 -5.956955188009836 0.9585898449575091 1.641209174573893 -2.4405474217252867;
    1.3179776848229032 1.0386077175091335 2.191677954355899 -0.5152838015317462 0.9730610147224401 3.0380696863621965 -1.3866542032979876;
];

% more globals
idq_ref_total=zeros(2*numSamples,trajectoryLength+1);
numWeights = numel(W1) + numel(W2) + numel(W3);

[J_total, e_hist_err, J_matix] = unrollTrajectoryFull(idq,trajectoryNumber,trajectoryLength,W3,W2,W1,flag,useShortcuts, use_idq);
writematrix(J_total, 'first_j_total.csv')
writematrix(e_hist_err, 'first_e_hist_err.csv')
writematrix(J_matix, 'first_j_matix.csv')

% testing with useShortcuts = true
useShortcuts = 1;
W2 = [
    1.440539114493213 -0.27253071839005777 0.5278868902219289 1.3712226806164334 2.25513928618451 1.3948446255239006 0.3449374254994523  1.3712226806164334 2.25513928618451 1.3948446255239006 0.3449374254994523;
    3.3789817246548357 0.19860814862310894 3.4592707210714577 -1.9110270294298068 0.2247519084049887 -1.0022103471763135 -0.4753830547340122 1.3712226806164334 2.25513928618451 1.3948446255239006 0.3449374254994523;
    -0.3590224302196529 1.2176551644649063 3.145578151429633 1.8631207326452712 3.708974043074285 -0.09608244193951347 1.5661350150973756  1.3712226806164334 2.25513928618451 1.3948446255239006 0.3449374254994523;
    4.031099888420788 -2.6851875859289094 -2.7498688647349647 2.7486598886675715 2.439552173754654 5.660170953027147 0.925728746264457 1.3712226806164334 2.25513928618451 1.3948446255239006 0.3449374254994523;
    -0.6221529426089921 0.7320648747641353 4.212370496471141 -4.0812165587839555 1.5473829764451348 -6.45653407631204 -0.817547511050558 1.3712226806164334 2.25513928618451 1.3948446255239006 0.3449374254994523;
    1.5225000509529556 -1.0360350040097754 1.7030720130819914 0.5344327232788693 0.6307629342167956 1.0930386330505284 0.07424903099042285 1.3712226806164334 2.25513928618451 1.3948446255239006 0.3449374254994523;
];

W3 = [
    -1.7111553944356483 -0.44719687703118854 -2.6145089128562864 -5.956955188009836 0.9585898449575091 1.641209174573893 -2.4405474217252867 -1.7111553944356483 -0.44719687703118854 -2.6145089128562864 -5.956955188009836 0.9585898449575091 1.641209174573893 -2.4405474217252867 0.9585898449575091 1.641209174573893 -2.4405474217252867;
    1.3179776848229032 1.0386077175091335 2.191677954355899 -0.5152838015317462 0.9730610147224401 3.0380696863621965 -1.3866542032979876  -1.7111553944356483 -0.44719687703118854 -2.6145089128562864 -5.956955188009836 0.9585898449575091 1.641209174573893 -2.4405474217252867 0.9585898449575091 1.641209174573893 -2.4405474217252867;
];

numWeights = numel(W1) + numel(W2) + numel(W3);

[J_total, e_hist_err, J_matix] = unrollTrajectoryFull(idq,trajectoryNumber,trajectoryLength,W3,W2,W1,flag,useShortcuts, use_idq);
writematrix(J_total, 'second_j_total.csv')
writematrix(e_hist_err, 'second_e_hist_err.csv')
writematrix(J_matix, 'second_j_matix.csv')
