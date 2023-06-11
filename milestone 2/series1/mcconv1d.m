% The Monte-Carlo method is applied to the computation of the first two
% moments of a random variable Y=U^{-a}, where U~Unif([0,1]) and
% a\in(-1,1). The mean is approximated using the sample mean and the
% variance by the sample variance. Confidence bounds of CLT and Chebyshev
% type are computed. 
%input: M- number of samples
%       a- coefficient in x^(-a)
         
%if flag is set to 1 CLT bounds are used
%if flag is set to 2 Chebyshev bounds with q=2 are used
%if flag is set to 3 Chebyshev bounds with q=1/a are used
%output: plots as in task 3a,3b
mmcconv1d(100000,-2/5)

%sample call : mcconv1d(100000,-2/3)
%              mcconv1d(100000,-2/5)


function mmcconv1d(M,a)
flag=2;
np = 10;
tic
delta=0.05;
q=1/a;
%Mean
exact = 1/(1-a);
%Variance
exact2=1/(1-2*a)-exact^2;
if exact2<0
    disp('Variance does not exist!!')
end
  Z = rand(1,M);
  X = 1./Z.^a;

  mean = cumsum(X)./(1:M);
  varest =cumsum((X-mean).*(X-mean))./(1:M);
  qmom=cumsum(abs(X-mean).^q./(1:M));
  err =abs(cumsum(X)./(1:M) - exact);
  
  if flag==1
  %Conf Inter. (CLT)       
  am = mean - erfinv(1-delta)*sqrt(2)*sqrt(varest./(1:M));
  bm = mean + erfinv(1-delta)*sqrt(2)*sqrt(varest./(1:M));
  elseif flag==2
  %Conf Inter. (Chebyshev)
  am = mean - delta^(-0.5)*sqrt(varest./(1:M));
  bm = mean + delta^(-0.5)*sqrt(varest./(1:M));
  elseif flag==3
  %Conf Inter. (Chebyshev) qth moment
  am = mean - delta^(-1/q)*qmom.^(1/q)./(1:M).^(1-1/q);
  bm = mean + delta^(-1/q)*qmom.^(1/q)./(1:M).^(1-1/q);
  else
      disp('Something is wrong!!')
  end
errp = err(np:np:end);  % plot only every np-th point
errvp=abs(varest(np:np:end)-exact2);
varest = varest(np:np:end);
if exact2<0
varest=varest*0;
end
meanp = mean(np:np:end);
amp = am(np:np:end);
bmp = bm(np:np:end);
Mp = (np:np:M);

toc

M1 = Mp(1); M2 = Mp(end);

subplot(3,1,1)

 loglog(Mp,errp,'b',[M1 M2],errp(end)*[(M1/M2)^-.5,1],'r',...
                  [M1 M2],errp(end)*[(M1/M2)^-(1/3),1],'g')
 legend('|mean-mean_m|','m^{-0.5}','m^{-1/3}')
             
subplot(3,1,2)            

 plot(Mp,meanp,'r',Mp,varest,'g', Mp,amp,'b', Mp,bmp, 'b')
 legend('Mean','Variance','Lower bound ','Upper bound ')

subplot(3,1,3)            

loglog(Mp,errvp,'b',[M1 M2],errvp(end)*[(M1/M2)^-.5,1],'r',...
                   [M1 M2],errvp(end)*[(M1/M2)^-(1/3),1],'g')
legend('|var-var_m|','m^{-0.5}','m^{-1/3}')
end
