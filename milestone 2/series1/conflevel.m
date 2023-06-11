%The Monte-Carlo method is applied to the computation of the first two
% moments of a random variable Y=U^{-a}, where U~Unif([0,1]) and
% a\in(-1,1). The mean is approximated using the sample mean and the
% variance by the sample variance. Confidence bounds of CLT, Chebyshev. 
% It is checked how many draws are in the
% bounds. The result is handeled to conflevel.m.
%if flag is set to 1 CLT bounds are used
%if flag is set to 2 Chebyshev bounds with q=2 are used
%if flag is set to 3 Chebyshev bounds with q=1/a are used

%Input: a- coefficient in a in f=x^-a
%       M- number of samples per iteration
%       K- number of iterations

%Output: plotssna as task 2c,2d,2e
cconflevel(-2/5,100000,1000)
%sample call : conflevel(2/3,100000,1000)
%              conflevel(2/5,100000,1000)

%The Monte-Carlo method is applied to the computation of the first two
% moments of a random variable Y=U^{-a}, where U~Unif([0,1]) and
% a\in(-1,1). The mean is approximated using the sample mean and the
% variance by the sample variance. Confidence bounds of CLT, Chebyshev and
% It is checked how many draws are in the
% bounds. The result is handeled to conflevel.m.
%input: M- number of samples
%       a- coefficient in x^(-a)
function y=mcc1d(M,a,flag)
q=1/a;
y = zeros(2,M);
exact = 1/(1-a);
delta=0.05;
  Z = rand(1,M);
  X = 1./Z.^a;
  %Mean
  mean = cumsum(X)./(1:M);
  %Stand. Devi.
  varest =cumsum((X-mean).*(X-mean))./(1:M);
  qmom=cumsum(abs(X-mean).^q./(1:M));
  %Error
  err = abs(cumsum(X)./(1:M) - exact);

 if flag==1
  %Conf Inter. (CLT)     
  am = mean - erfinv(1-delta)*sqrt(2)*sqrt(varest./(1:M));
  bm = mean + erfinv(1-delta)*sqrt(2)*sqrt(varest./(1:M));
 elseif flag==2
  %Conf Inter. (Chebyshev)
   am = mean - delta^(-1/2)*sqrt(varest./(1:M));
   bm = mean + delta^(-1/2)*sqrt(varest./(1:M));
 elseif flag==3
   %Conf Inter. (Chebyshev) qth moment
    am = mean - delta^(-1/q)*qmom.^(1/q)./(1:M).^(1-1/q);
    bm = mean + delta^(-1/q)*qmom.^(1/q)./(1:M).^(1-1/q);
 %elseif flag==4
 %   %Conf Inter. (Fishman bounds)
 %   M
 %   [am,bm]=DFI(mean(end),M,delta); 
 else 
     disp('Something is wrong!!')
 end
 
     %Samples in the bounds
  y(1,:) = (bm>exact).*(am<exact); % counts whether exact mean is in [am,bm].
  %
  y(2,:) = err;
 
end


function []=cconflevel(a,M,K)
flag=1;
y = zeros(2,M); % y(1,:) measures the error, y(2,:) is for confidence level.
if a>0.5
    disp('Variance does not exist!!')
end

for k=1:K
   y = y+mcc1d(M,a,flag);
end
np = 10;
Mp = (np:np:M);
yp = y(:,np:np:end)/K;


M1 = Mp(1); M2 = Mp(end);
subplot(2,1,1)
loglog(Mp,yp(2,:),'b',[M1 M2],yp(2,end)*[(M1/M2)^-.5,1],'r',...
                   [M1 M2],yp(2,end)*[(M1/M2)^-(1/3),1],'g') 
 legend('|mu-mu_m|','m^{-0.5}','m^{-1/3}')
 subplot(2,1,2)            
 plot(Mp,yp(1,:))
 legend('Confidence level')
end