%function [W] = myfun(X1,Y1,c1,c3,c4,c5,c6,c7)
function [Pre_labels_train ,Pre_labels_test,time] = myfunupdated(X1,Y, X2,Y2, par)
c1=par.c1;
c3=par.c3;
c4=par.c4;
c5=par.c5;
c6=par.c6;
c7=par.c7;
% c8=par.c8;

[N2,M1]=size(Y);% M1 classes
% N2 samples. D dim
[N2,D]=size(X1);%training
[N1,D]=size(X2);%testing

SI = exp(-squareform(0.5*pdist(X1)).^2);
delta = diag(sum(SI, 2));
delta = (delta)^(-1/2);
Ls = diag(sum(SI, 2)) - SI;
Ls = delta*Ls*delta;
%

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
for i=1:10
A=(2*c1*I + c3*Ls + c4*U)^-1;

B=(c1*X1*W+c4*U*Y);


F=A*B;%Update F





%  ======Update S fix W======
% sig = 0.7 ;
sig=(1/N2^2)*(norm(X1)^2);
RI = (eye(N2)-evalkernel(X1,X1,'rbf',sig))+eye(N2)*10^100;% acc to eqn 5 diagonal are infinity
YY=X1*W;

% S=inv(c5*YY*YY'+c6*RI)*YY*YY'
%  S=c8*Ls+((c5*YY*YY'+c6*RI)\YY)*YY';
 S=0.5*((YY*YY'+(c6\c5)*RI)\YY)*YY';
end

% S = eye(N2);
% 
% L=RI;
% for j=1:N2
%     R=diag(L(:,j));
% %     S(:,j)=((2*c5*Y*Y'+c6*R)^-1)*c5*Y*(Y(j,:))';
%      S(:,j) = ((c5*(YY*YY') + c6*R)\ YY)*YY(j,:)';YY
% end



% =======update W ============

M=I-S;

W=c7*((c7*X1'*X1+c1*Dw+c6*X1'*X1+c4*X1'*M*M'*X1)^-1)*X1'*F;
% time=toc;
Pre_labels_train = X1*W;
Pre_labels_train = sign(Pre_labels_train);
Pre_labels_test = X2*W;
Pre_labels_test = sign(Pre_labels_test);
time=toc;
end


%[N1,D1]=size(X1);
%B=ones(N1,1);
%X1=cat(2,X1,B)
%X = normalize(X1,'range');

% L=squareform(sig*(pdist(X).^2));
% 
%SI = exp(-squareform(0.5*pdist(X)).^2); %Try GAUSSIAN

% L=RI;
% for j=1:N2
%     R=diag(L(:,j));
%     %S(:,j)=((2*c5*Y*Y'+c6*R)^-1)*c5*Y*(Y(j,:))';
%     ss(:,j) = (c5*YY*YY' + c6*R) \ ( YY*(YY(j,:)') );
% end
