%
% This file is needed by "net_action.m" to calculate the derivatives through the NN controller.
%
function y=exdiag(x)
% This function is an extension of built-in function diag, which implement
% the following functions, e.g.
%
% exdiag([1 2 3;4 5 6])
%
% ans =
%
%      1     2     3     0     0     0
%      0     0     0     4     5     6
[m,n]=size(x);
y=zeros(m,m*n);
for i=1:m
    y(i,n*(i-1)+1:n*i)=x(i,:);
end
% y=sparse(y);