%Function generates 10 sample paths of an geometric Brownian motion with
%drift

function[]= ex43a
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

