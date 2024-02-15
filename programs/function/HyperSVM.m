
clc;
clear all;
close all;
A = importdata("avocado.csv");

% load data

data = A.data;


%k(x,y) = exp(-?*||x-y||^2)
d =dist(data,"euclidean");

e =( d)/2;

W = exp(-e);
m = mean(data);

% eign value and eigen vector
D =(mean(W,2));
L = D-W;
[evector , evalue] = eig(L);
evalue = diag(evalue);
evalue =  sortrows(evalue)
KNN=5;
H =Hyper(data,KNN);
de = (sum(H,2));
dv = (sum(H,1));
Dv =  diag(dv);
De = diag(de);
Y = inv(De);
weight = [1/de*(de-1)]* W/m

W1 = diag(weight);

%laplacian matrix of the hypergraph
T = H*H'*Y*W1



%r1 = ( sum(sum(evalue(11:12,:))) / sum(sum(evalue(1:12,:))) ) * 100

%U = evector(:,[ 11 12 ])  % Taking higest eigenvalue eigenvector 






    



