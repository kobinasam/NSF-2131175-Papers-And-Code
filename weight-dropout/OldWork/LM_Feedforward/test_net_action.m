
global Gain
Gain = 20;
Gain2 = 100;
Gain3 = 100;
costTermPower = 1;
integral_decay_constant = 1;

numHids1=3;
numOutputs=1;
numInputs=2;
useShortcuts=0;

w1 = [
    0.10511860249074981, -0.8691958077685074, 3.9107265744512154;
    0.8052531277712192, 0.11608271973910032, 5.081202415079452;
    0.14211713490656408, 0.2723755030710997, 4.040974596086221;
];


w3 = [
    -1.7111553944356483, -0.44719687703118854, -2.6145089128562864, -5.956955188009836;
];

idq = [1; 2];

out = net_action(idq, w1, w3);

[Dnet_Dw,~]=net_action_backpropagate(idq,w1,w3);