NN=1000;
idq=rand(2,1);
idq_his=zeros(2,NN);
for i=1:NN
    idq=a*idq+b*([1;0]*Vmax-Vdq);
    idq_his(:,i)=idq;
end
idq
plot(idq_his')  

