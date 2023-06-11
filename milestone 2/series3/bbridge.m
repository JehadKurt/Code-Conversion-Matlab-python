%%%Pathsimulation via Brownian Bridge, path refinement
%%%Parameters T=final time, L=number of refinements N0=number of grid
%%%points on the first level
T = 1;                            % final time
L= 8;
N0 =10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = N0*2^L;
EMX = zeros(L+1, N+1);
x0 =0;
h= T/N;
B = [zeros(1,1);sqrt(T/N0)*cumsum(randn(N0,1))];% path of Brownian motion at crudest level
EMX(1, 1:2^L:end) = B;
t = (T*(0:N)'/N);
subplot(3,3,1); plot((T*(0:N0)'/N0), B); 
 xlabel('time')
   ylabel('value')
  % the brownian bridge simulations.
  for i=2:L+1             %% refinement of paths via Brownian bridge. 
      p = 2^(L-i+1);
      EMX(i,:)=EMX(i-1,:);
      h= T/(N0*2^(i-1));
      len=N/p/2;
     EMX(i, [p+1:2*p: N+1-p])= .5*(EMX(i,[1:2*p: N+1-2*p])+EMX(i,[2*p+1:2*p: N+1]))+sqrt(h./2)*randn(1,len);
   subplot(3,3,i); plot(t(1:p:N), EMX(i, 1:p:N)'); 
   xlabel('time')
   ylabel('value')
  end
  