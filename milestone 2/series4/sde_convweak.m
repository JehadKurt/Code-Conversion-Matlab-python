function sde_convweak
% SDE X_t = f(X_t,t)dt + g(X_t,t)dB_t
%     X_0 = x0; t in [0,T] is solved using the EM method
%
% all paths are computed at the same time
N0=10;                            % number of steps on coarsest level
L=4;                              % number of refinement steps
M=5*10^4;                           % number of samples 
itera=10;
T = 1;                            % final time
x0 = 1;                           % initial condition
f=@(x,t)(-sin(x).*(cos(x)).^3);   % coeffients of the SDE
g=@(x,t)cos(x).^2;
G=@(x) (max(x-1.1,0));            



NL = N0*2^L;
B = brownp(T,NL,M);               % paths of Brownian motion on finest level
YT = zeros(L+1,M);                % values of X_T for h=T/(N*2^l), l=0,...,L, all m
Erro1=zeros(L+1,1);
Erro2=zeros(L+1,1);
tic
for it=1:itera
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
    Y(j+1,:) = x + (-sin(x).*(cos(x)).^3)*h + (cos(x).^2).*dB;
  end
  YT(l+1,:) = Y(N+1,:);           % values of X_T
end
Z = G(YT);                        % compute payoffs


%  weak convergence :
Ym=mean(Z,2);
Ymex = mean(G(atan(randn(1,1e7)*sqrt(T)+tan(x0))));%    sample means for Y
Yme = abs(Ym-Ymex);       %    error in means for weak convergence
Erro1=Erro1+Yme;
Erro2=Erro2+Yme.^2;
end
Erro1=Erro1/itera;
Erro2=(Erro2/itera).^0.5;
Lp = L;                       % plot results for l=0,...,Lp 
                          
hv = T./(N*2.^(0:Lp)');
hL = hv(end); p = hv(1)/hL;

figure(1); loglog(hv,Erro1,'-gx',hv,Erro2,'-ro',hL*[1 p],Erro1(end)*[1 p^1.0],hL*[1 p],Erro2(end)*[1 p^1.0]); 
legend('weak error in L^1', 'weak error in L^2', ' h^{1}',' h^{1}'); grid on
xlabel('step size $h$')
ylabel('error')

% compute convergence rate
p = polyfit(log(hv),log(Erro2),1);
disp('Weak rate of convergence L^2')
disp(p(1));

p = polyfit(log(hv),log(Erro1),1);
disp('Weak rate of convergence L^1')
disp(p(1));

toc
