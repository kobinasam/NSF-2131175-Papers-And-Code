


W1=[0.081472368639318   0.027849821886705   0.095716694824295   0.079220732955955   0.067873515485777
    0.090579193707562   0.054688151920498   0.048537564872284   0.095949242639290   0.075774013057833
    0.012698681629351   0.095750683543430   0.080028046888880   0.065574069915659   0.074313246812492
    0.091337585613902   0.096488853519928   0.014188633862722   0.003571167857419   0.039222701953417
    0.063235924622541   0.015761308167755   0.042176128262628   0.084912930586878   0.065547789017756
    0.009754040499941   0.097059278176062   0.091573552518907   0.093399324775755   0.017118668781156];

W2=[0.012649986532930   0.031747977514944   0.055573794271939   0.055778896675488   0.025779225057201   0.040218398522248   0.087111112191539
    0.013430330431357   0.031642899914629   0.018443366775765   0.031342898993659   0.039679931863314   0.062067194719958   0.035077674488589
    0.009859409271100   0.021756330942282   0.021203084253232   0.016620356290215   0.007399476957694   0.015436980547927   0.068553570874754
    0.014202724843193   0.025104184601574   0.007734680811268   0.062249725927990   0.068409606696201   0.038134520444447   0.029414863376785
    0.016825129849153   0.089292240528598   0.091380041077957   0.098793473495250   0.040238833269616   0.016113397184936   0.053062930385689
    0.019624892225696   0.070322322455629   0.070671521769693   0.017043202305688   0.098283520139395   0.075811243132742   0.083242338628518];

W3=[0.002053577465818   0.065369988900825   0.016351236852753   0.079465788538875   0.044003559576025   0.075194639386745   0.006418708739190
    0.092367561262041   0.093261357204856   0.092109725589220   0.057739419670665   0.025761373671244   0.022866948210550   0.076732951077657];

weights = {W1, W2, W3};

% MANUAL TEST
idq = [1; 2];
idqRef = [1.1; 2.1];
histErr = [1.3; 1.4];
gain1 = 1;
gain2 = 2;
gain3 = 3;
use_idq = false;

input0A=(idq)/gain1;
output0A=tanh(input0A);
input0B=(idq(1:2)-idqRef)/gain2; %% Note: error term = idq(1:2)-idq_ref
output0B=tanh(input0B);
input0C=histErr/gain3;
output0C=tanh(input0C);

% the first hidden layer
if use_idq==1
    input1=[output0A; output0B; output0C; -1]; % input1=[output0A; output0B; -1];
else
    input1=[output0B; output0C; -1];
end

sum1=W1*input1;
o1=tanh(sum1);

% the second hidden layer
input2=[o1; -1];
sum2=W2*input2;
o2=tanh(sum2);

% the output layer
input3=[o2; -1];
sum3=W3*input3;
o3=tanh(sum3);

hiddenLayers = [
    Layer(neurons=6, activation=Tanh),  ... 
    Layer(neurons=6, activation=Tanh)   ...
];

model=Rnncontroller(            ...
    hiddenLayers=hiddenLayers,  ...
    gain1=gain1,                ...
    gain2=gain2,                ...
    gain3=gain3,                ...
    initWeights=weights,        ...
    useIdq=false                ...
);

[modelInputs, modelOutputs] = model.netAction(idq, idqRef, histErr);

% WE EXPECT THE MODEL INPUTS / OUTPUTS TO BE IDENTICAL TO THE MANUALLY COMPUTED input1, input2, input3, o1, o2, o3
all(input1 == modelInputs{1})
all(input2 == modelInputs{2})
all(input3 == modelInputs{3})
all(o1 == modelOutputs{1})
all(o2 == modelOutputs{2})
all(o3 == modelOutputs{3})
