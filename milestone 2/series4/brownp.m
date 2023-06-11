function x = brownp(T,N,M)
% x = brownp(T,N)
% x = brownp(T,N,M)
% find M paths of Brownian motion, default M=1
% each column of x contains values at t=T/N, 2*T/N, ... , T
if nargin==2
  M=1;
end
x = [zeros(1,M);sqrt(T/N)*cumsum(randn(N,M))];
