%function [W] = myfun(X1,Y1,c1,c3,c4,c5,c6,c7)
function [Pre_labels_train ,Pre_labels_test,time,obj] = myfunupdatedHyper(X1,Y, X2,Y2, par)
c1=par.c1;
c2=par.c2;
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
knn=floor(0.2*N2);
the=0.6;
%SI=Hyper(X1,knn);
% SI=HyperedgeHeatKernel(X1);
SI=Hyper_incident_Matrix(X1,the);
%Ls=HypLap(SI,'Saito');
 Ls=HypLap(SI,'Zhou');
%Ls=HypLap(SI,'Rod');

% SI = exp(-squareform(0.5*pdist(X1)).^2);
% delta = diag(sum(SI, 2));
% delta = (delta)^(-1/2);
% Ls = diag(sum(SI, 2)) - SI;
% Ls = delta*Ls*delta;
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

 S=0.5*((YY*YY'+(c6\c5)*RI)\YY)*YY';

% =======update W ============

M=I-S;
II=eye(M1,M1);
W=c7*((c7*X1'*X1+c1*Dw+c6*X1'*X1+c4*X1'*M*M'*X1)^-1)*X1'*F;
% time=toc;
 temp1=c1*(norm((X1*W-F),'FRO'))^2;
 temp3=c2*sum(sqrt(sum(W.^2, 2)));
 temp4=c3*trace(F'*Ls*F);
temp5=c4*trace((F-Y)'*U*(F-Y));
 temp2=c5*(norm(W'*(X1'-X1'*S)))^2;
 temp6=c6*(norm(RI.*S))^2;
 temp8=c7*trace(W'*X1'*X1*W-II);
obj(i)=temp1+temp2+temp3+temp4+temp5+temp6+temp8;
end
Pre_labels_train = X1*W;
Pre_labels_train = sign(Pre_labels_train);
Pre_labels_test = X2*W;
Pre_labels_test = sign(Pre_labels_test);
time=toc;
end



 
 


