function [Xnew]= gllefun(Kdim,X)
Knbh=Kdim;
[N,D] = size(X);
fprintf(1,'LLE running on %d points in %d dimensions\n',N,D);
% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS
fprintf(1,'-->Finding %d nearest neighbours.\n',Knbh);

X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;

[sorted,index] = sort(distance);
neighborhood = index(2:(1+Knbh),:);
clc


% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
fprintf(1,'-->Solving for reconstruction weights.\n');

if(Knbh>D)
    fprintf(1,'   [note: K>D; regularization will be used]\n');
    tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
    tol=0;
end
sig=0.1;
W = zeros(Knbh,N);
for ii=1:N
    z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,Knbh); % shift ith pt to origin
    C = z'*z;                                        % local covariance
    C = C + eye(Knbh,Knbh)*tol*trace(C);                   % regularlization (K>D)
    W(:,ii) = C\ones(Knbh,1);                           % solve Cw=1
    W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
end;
Wnew=zeros(N,N);
for kk=1:Knbh
    for jj=1:N
        index=neighborhood(kk,jj);
        Wnew(index,jj)=W(kk,jj);
    end
end
%P=X-X*Wnew';
P=X-X*Wnew;
for i=1:N
    for j=1:N
        if ismember(i,neighborhood(:,j))||ismember(j,neighborhood(:,i))
            %s(i,j)=exp-(norm(data(i,:)-data(j,:)*0.3)
            %S(i,j)=exp(-(norm(X(:,i)-X(:,j))/sig^2));
            S(i,j)=exp(-sig*(norm(X(:,i)-X(:,j))));
        end
    end
end
rng(11,'twister')
%B=zeros(Kdim,D);
%inx1=randi([1,D],D,1);
%inx=unique(inx1);
%inx=inx(1:Kdim);
%for i=1:length(inx)
%B(i,inx(i))=1;
%end
%B1=B;
B=ones(Kdim,N);
B=(1/D).*B;
J3=P*P';
J3P=((abs(J3)+J3)/2);
J3N=((abs(J3)-J3)/2);
a1=sum(S,2);
D1=diag(a1);
L=D1-S;
%sig=0.7;
alp=.5;
lam=1;
mu=10;
Q=B*P;
%sigma=eye(Kdim,N);
sigma=ones(Kdim,N);
for ll=1:2
    J1=X*L*X';
    J1P=((abs(J1)+J1)/2);
    J1N=((abs(J1)-J1)/2);
    J4=sigma*P';
    J4P=((abs(J4)+J4)/2);
    J4N=((abs(J4)-J4)/2);
    J2=Q*P';
    J2P=((abs(J2)+J2)/2);
    J2N=((abs(J2)-J2)/2);
    Numo=2*alp.*B*J1N+2*mu.*B+lam.*J2P+lam.*B*J3N+J4P;
    Deno=2*alp.*B*J1P+2*mu.*B*B'*B+lam.*J2N+lam.*B*J3P+J4N;
    Div=((Numo./Deno).^0.5);
    B=B.*Div;
    Na=B*P+sigma;
    for i=1:Kdim
        for j=1:N
            Q(i,j)=((sign(Na(i,j)))/lam)*(max(0,lam*abs(Na(i,j))-1));
        end
    end
    sigma=sigma+lam.*(Q-B*P);
end

%for
 b=vecnorm(B,2,1);

 [ind,~]=sort(b,'descend');
%time=toc;
%X=X';
for j=1:Kdim
    Xnew(j,:)=X(ind(j),:);  
end 
data=Xnew;
end 



