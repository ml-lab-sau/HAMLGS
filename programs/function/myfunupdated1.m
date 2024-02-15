%function [W] = myfun(X1,Y1,c1,c3,c4,c5,c6,c7)
function [Pre_labels_train ,Pre_labels_test,time,W] = myfunupdated1(X1,Y, X2,Y2, c1,c3,c4,c5,c6,c7,c8)
% N2 samples. D dim
tic;
[N2,D]=size(X1);%training
[N1,D]=size(X2);%testing
B1=ones(N2,1);
B2=ones(N1,1);
X1=cat(2,X1,B1);
X2=cat(2,X2,B2);
[N2,M1]=size(Y);
D=D+1;
sig=(1/N2^2)*(norm(X1)^2);
% sig=0.7;


% SI = exp(-squareform(0.5*pdist(X1)).^2);
% delta = diag(sum(SI, 2));
% delta = (delta)^(-1/2);
% Ls = diag(sum(SI, 2)) - SI;
% Ls = delta*Ls*delta;
% sig = 0.7 ;
Ls = evalkernel(X1,X1,'rbf',sig);
L = diag(sum(Ls, 2)) - Ls;

U=zeros(N2,N2);
for i=1:N2
    if(Y(i,1)~=0)
        U(i,i)=100;
    end
end


W=randn(D,M1);
Dw=zeros(D,D);
for ii=1:D
    Dw(ii,ii)=1/(2*norm(W(ii,:)));
end


I=eye(N2,N2);

tic;

A=(2*c1*I + c3*Ls + c4*U)^-1;

B=(c1*X1*W+c4*U*Y);


F=A*B;%Update F





%  ======Update S fix W======

 RI = evalkernel(X1,X1,'rbf',sig) +eye(N2)*10^100;
% for i=1:5
 YY=X1*W;

%  S = eye(N2);

 % =======update W ============

S=((c5*YY*YY'+c6*RI)\YY)*YY';
M=I-S;
W=c7*((c7*X1'*X1+c1*Dw+c6*X1'*X1+c4*X1'*M*M'*X1)^-1)*X1'*F;
% end
time=toc;
Pre_labels_train = X1*W;
Pre_labels_train = sign(Pre_labels_train);
Pre_labels_test = X2*W;
Pre_labels_test = sign(Pre_labels_test);
%time=toc;
end


