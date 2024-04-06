function y=exdiag(x)
[m,n]=size(x);
y=zeros(m,m*n);
for i=1:m
    y(i,n*(i-1)+1:n*i)=x(i,:);
end