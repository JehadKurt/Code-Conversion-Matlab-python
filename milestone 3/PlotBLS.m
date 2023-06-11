% PlotBLS.m
S0 = 30:1:70;
X = 50;
r = 0.08;
sigma = 0.4;
for T=2:-0.25:0
   plot(S0,blsprice(S0,X,r,T,sigma));
   hold on;
end
axis([30 70 -5 35]);
grid on