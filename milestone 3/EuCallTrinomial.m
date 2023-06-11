function price = EuCallTrinomial(S0,K,r,T,sigma,N,deltaX)
% Precompute invariant quantities
deltaT = T/N;
nu = r - 0.5*sigma^2;
discount = exp(-r*deltaT);
p_u = discount*0.5*((sigma^2*deltaT+nu^2*deltaT^2)/deltaX^2 + ...
    nu*deltaT/deltaX);
p_m = discount*(1 - (sigma^2*deltaT+nu^2*deltaT^2)/deltaX^2); 
p_d = discount*0.5*((sigma^2*deltaT+nu^2*deltaT^2)/deltaX^2 - ...
    nu*deltaT/deltaX);
% set up S values (at maturity)
Svals = zeros(2*N+1,1);
Svals(1) = S0*exp(-N*deltaX);
exp_dX = exp(deltaX);
for j=2:2*N+1
    Svals(j) = exp_dX*Svals(j-1);
end
% set up lattice and terminal values
Cvals = zeros(2*N+1,2);
t = mod(N,2)+1;
for j=1:2*N+1
    Cvals(j,t) = max(Svals(j)-K,0);
end
for t=N-1:-1:0;
    know = mod(t,2)+1;
    knext = mod(t+1,2)+1;
    for j = N-t+1:N+t+1
        Cvals(j,know) = p_d*Cvals(j-1,knext)+p_m*Cvals(j,knext)+p_u*Cvals(j+1,knext);
    end
end
price = Cvals(N+1,1);

