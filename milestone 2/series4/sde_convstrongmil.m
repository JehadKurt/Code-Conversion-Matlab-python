function sde_convstrongmil
% SDE X_t = f(X_t,t)dt + g(X_t,t)dB_t
%     X_0 = x0; t in [0,T] is solved using the EM method
%
% all paths are computed at the same time
N0=10;                            % number of steps on coarsest level
L=6;                              % number of refinement steps
M=10^4;                           % number of samples 

T = 1;                            % final time
x0 = 1;                           % initial condition
f=@(x,t)(-sin(x).*(cos(x)).^3);   % coeffients of the SDE
g=@(x,t)cos(x).^2;
hfu=@(x,t)2*cos(x).^3.*(-sin(x));

NL = N0*2^L;
B = brownp(T,NL,M);               % paths of Brownian motion on finest level
YT = zeros(L+1,M);                % values of X_T for h=T/(N*2^l), l=0,...,L, all m
tic
for l=0:L                         % for N = N0, N0*2, ... , N0*2^L
  N = N0*2^l;
  p = 2^(L-l);                    % p = NL/N
  h = T/N; 
  Y = zeros(N+1,M);
  Y(1,:) = x0;
  for j=1:N                       % perform N steps of E-M method:
    dB = B(1+j*p,:) - B(1+(j-1)*p,:);
    t = j*h;
    x=Y(j,:);
Y(j+1,:) = x + f(x,t)*h + g(x,t).*dB;
Y(j+1,:) = x + f(x,t)*h + g(x,t).*dB+0.5*hfu(x,t).*(dB.^2-h);
  end
  YT(l+1,:) = Y(N+1,:);           % values of X_T
end

%strong convergence :              
exact=atan(B(end,:)+tan(x0));
YTe = abs(YT-repmat(exact,L+1,1));   %    errors for XT
YTem = mean(YTe,2);             %    mean errors for strong convergence
YTem2 = sum(YTe.^2,2).^0.5;         %    mean square errors for strong convergence

Lp = L;                       % plot results for l=0,...,Lp 
                          
hv = T./(N*2.^(0:Lp)');
hL = hv(end); p = hv(1)/hL;

%figure(1); loglog(hv,YTem2,'-gx',hv,YTem,'-ro',hL*[1 p],YTem(end)*[1 p^.5],hL*[1 p],YTem2(end)*[1 p^.5]);
figure(1); loglog(hv,YTem2,'-gx',hv,YTem,'-ro',hL*[1 p],YTem(end)*[1 p^1],hL*[1 p],YTem2(end)*[1 p^1]); 
legend('strong error in L^1', 'strong error in L^2', ' h^{1}',' h^{1}'); grid on
xlabel('step size $h$')
ylabel('error')
% compute convergence rate
p = polyfit(log(hv),log(YTem),1);
disp('Strong rate of convergence')
disp(p(1));

toc
