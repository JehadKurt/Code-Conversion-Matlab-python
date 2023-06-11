%M samples of a geometric Brownian motion at t=T are generated. The
%expectation E((K-S_1)_+) is apprximated. Confidence bounds using CLT are
%computed.

clear all
close all

M = 10^6;
T= 1;
SO = 10;
sigma = .5;            %volatility
K = 11;                % strike price.
X = sqrt(T)*randn(1,M);

S = SO*exp(sigma*X-.5*T*(sigma^2)); %simulated stock prices. 
HS = max(K-S,0);% calcuation of payoff.
price = mean(HS);

% 95% confidence intervals%
AM= price-1.96*sqrt((var(HS))/M); %based on CLT.
BM= price+1.96*sqrt((var(HS))/M); %%

disp('Estimated value: ')
disp(price)
disp('Confidence interval: ')
disp([AM,BM])
ex43aa

function[]= ex43aa
t=1;
N=10^5;
M=10;
c=1;
sigma=0.5;
x = [zeros(1,M);sqrt(t/N)*cumsum(randn(N,M))];
drift=c*repmat((0:t/N:t)',1,M);
x=sigma*x+drift;
x=exp(x);
plot(0:t/N:1,[x,exp(drift(:,1))])
xlabel('Time')
ylabel('Value')
end