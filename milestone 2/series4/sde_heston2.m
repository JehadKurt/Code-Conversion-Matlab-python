function sde_heston2
% solve SDE for Heston model
% using Euler-Maruyama with N0,2*N0,...,2^L*N0 time steps
% compute strong and weak rates using overkill solution with 2^(L+extra)*N0 time steps
% all paths generated at the same time


N0=10;                            % number of steps on coarsest level
L=5;                              % number of refinement steps
M=10^4;                           % number of samples 
T = 1;                            % final time
x0 = 10;                          % initial condition for X_t
v0 = .5;                          % initial value for volatility
extra = 3;                        % extra levels of refinement for overkill solution
xi = .25;
theta = .5;
r=0.05;
kappa = 2;
G=@(x)max(11-x,0);

%%%%%%%%%%
tic
Le = L + extra;
Ne = N0*2^Le; 
  BI = brownp(T,Ne,M);        	  % path of Brownian motion on finest level
  BII = brownp(T,Ne,M);
  YT = zeros(L+2,M);         	  % values of X_T for h=T/(N*2^l), l=0,...,Le
Lv = [0:L,Le]; 
%loop over levels
for l=1:L+2 % l=0,...,L,Le				     
    le=Lv(l);
    N = N0*2^le;
    p = 2^(Le-le);            	  % p = Ne/N					     
    h = T/N; 
    x = x0;
    v=v0;
    %loop over increments
    for j=1:N                	  % perform N steps of E-M method
      dBI = BI(1+j*p,:) - BI(1+(j-1)*p,:);  % Brown process increment
      dBII = BII(1+j*p,:) - BII(1+(j-1)*p,:);  % Brown process increment
      x = x + r*x*h + ((abs(v)).^.5).*x.*dBI;               
      v = v+kappa*(theta-v)*h+xi*((abs(v)).^.5).*dBII;
    end
    %
    YT(l,:) = x;                  % values of X_T
end
%
Ys = G(YT);                     % compute payoffs
YTe = abs(YT(1:L+1,:) - repmat(YT(L+2,:),L+1,1));% errors for XT compared to overkill solution
YTem = mean(YTe,2);                    % mean errors for strong convergence
YTem2 = sum(YTe.^2,2).^0.5;                    % mean errors for strong convergence
Ym = mean(Ys,2);                        % sample means for Y
est_var=var(Ys(1:end-1,:),1,2);

AM= Ym(1:end-1)-1.96*sqrt(est_var/(M)); %based on CLT.
BM= Ym(1:end-1)+1.96*sqrt(est_var/(M)); % 
hv = T./(N0*2.^(0:L)');           % vector of h values
hL = hv(end); p = hv(1)/hL;

disp('CLT confidence interval')
disp([AM BM])
%Plots
figure(1); loglog(hv,YTem2,'-gx',hv,YTem,'-ro',hL*[1 p],YTem(end)*[1 p^.5],hL*[1 p],YTem2(end)*[1 p^.5]);
legend('strong error in L^1', 'strong error in L^2', ' h^{1/2}',' h^{1/2}'); grid on
xlabel('step size $h$')
ylabel('error')
toc
