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
 