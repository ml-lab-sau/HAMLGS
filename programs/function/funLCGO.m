function [Xnew,time] = funLCGO(X1,par)
al=par.al;
Bt=par.bt;
X=X1;
[N,D]=size(X);
% =Y1;
d1=par.k;
%output P and S
%parameter alpha and Beta
% Compute the similarity matrix W and locality adaptor vector Ri (i = 1,...,n)
% Repeat
% Update S•i (i = 1,....,n) according to Eq. (14)
% Update P using Eq. (17)
% until stop criteria is met
% yi = PT xi and Y = PTX.
%  sig=.5;
tic
P=ones(D,d1);
YY=P'*X';
% al=.1; Bt=.1;
sig=(1/N^2)*norm(X)^2;
R=evalkernel(X,X,'rbfp',sig)+eye(N)*10^100;
W = evalkernel(X,X,'rbf',sig);
%Update S
%S•i = (YTY + αEi + βL)−1YT yi
L = diag(sum(W, 2)) - W;
I=eye(N);
% for i=1:N
%     y(:,i)=P'*X(:,i);
%     E=diag(W(:,i));
%     S(:,i)= (YY'*YY+al*E+Bt*L)\YY'*y(:,i);
% end 
S=ones(N,N);

%  for j=1:5
while((((norm(P'*(X'-X'*S)))^2+al*S'*R*S+Bt*trace(S'*L*S))>=0)& (j<=10))
S=(YY'*YY+al*W+Bt*L)\YY'*YY;
M= X'*(I-S-S'+S'*S)*X;
%  end
C = X' * X;
[V,D]=eig(M,C);
[d,ind] = sort(diag(D));

P=V(:,ind(1:d1));
YY = P'*X';

end
 %disp("P");

time=toc;
Xnew=P'*X';
Xnew=Xnew';
