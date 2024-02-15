clc
clear all
close all
x1= [1 2 3 4 5 6 7 8 9 10];
y1= [4 2 3 1.7 1 1.2 1.5 1.9 2.3 2.7];
error=[];
error2=[];
for i=1:length(x1)
a=2;
x=[];
y=[];
for ij=1:length(x1)
    if(ij~=i)
        x=[x x1(ij)];
        y=[y y1(ij)];
    end
end
coeff=polyfit(x,y,a);
temp=a;
c=0;
t1=x1(i);
for j=1:length(coeff)
c=c+coeff(j)*(t1^temp);
temp=temp-1;
end
temp=a;
exa=polyval(c,t1);
error=[error abs(y1(i)-c)];
error2=[error2 abs(y1(i)-exa)];
c=0;
end