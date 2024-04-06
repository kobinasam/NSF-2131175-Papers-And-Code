function  [Dnet_Dw,Dnet_Didq]=net_action_backpropagate(idq,W1,W3)


%%
global numWeights numHids1   ;

    global  Gain    ;

   output0A=(idq)/Gain;
    input1=[output0A; -1];

    sum1=W1*input1; o1=tanh(sum1);     % first hidden layer outputs
    input3=[o1; -1];

    sum3=W3*input3; out=tanh(sum3);         % network outputs

%% compute Dnet_Dw
[m1,n1]=size(W1); [m3,n3]=size(W3);
Dnet_Dw=zeros(size(out,1),numWeights);
Dnet_Didq=zeros(2,2);

% second layer
Dnet_Dw3=(1-out.^2)*input3';
Dnet_Dw(1,1:n3)=Dnet_Dw3(1,:);


% first layer
dW3=diag(1-out.^2)*W3;

for i=1:m3
    Dnet_Dw2=((dW3(i,1:numHids1)').*(1-o1.^2))*input1';
    Dnet_Dw2=Dnet_Dw2';Dnet_Dw2=Dnet_Dw2(:);Dnet_Dw2=Dnet_Dw2';
    Dnet_Dw(i,n3*m3+1:n3*m3+n1*m1)=Dnet_Dw2;
    clear Dnet_Dw2
end
