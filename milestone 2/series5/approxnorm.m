%Function generates n samples of a normally distibuted random variable. Two
%methods are used: a CLT approximation with 12 factors and the rand routine
%in MATLAB. The two methods are compared via qqplots and histograms
% Input:  n    number of samples
approxnorm1(100)

function[]=approxnorm1(n)
approx_1=sum(rand(n,12),2)-6;
approx_2=randn(n,1);
figure(1)
qqplot(approx_1)
title('CLT approximation')
figure(2)
qqplot(approx_2)
title('randn sample')
figure(3)
hist(approx_1,-5:0.1:5)
title('CLT approximation')
figure(4)
hist(approx_2,-5:0.1:5)
title('randn sample')
end