% AsianMC.m
function [P,CI] = AsianMC(S0,X,r,T,sigma,NSamples,NRepl)
Payoff = zeros(NRepl,1);
for i=1:NRepl
   Path=AssetPaths(S0,r,sigma,T,NSamples,1);
   Payoff(i) = max(0, (prod(Path(2:(NSamples+1))))^(1/NSamples) - X);
end
[P,aux,CI] = normfit( exp(-r*T) * Payoff);