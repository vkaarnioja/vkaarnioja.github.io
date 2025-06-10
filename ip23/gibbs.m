% A simple Gibbs sampler algorithm

clear all; close all;
p = @(x,y) exp(-10*(x.^2-y).^2-(y-1/4).^4);
x = [0,0];
nsamples = 5000;
samples = [];
xx = linspace(-2,2,10000);
for jj = 2:nsamples
    u = rand;
    P = cumsum(p(xx,x(2)));
    P = P/P(end);
    ind = find(u <= P,1,'first');
    x(1) = xx(ind);
    u = rand;
    P = cumsum(p(x(1),xx));
    P = P/P(end);
    ind = find(u <= P,1,'first');
    x(2) = xx(ind);
    samples = [samples;x];
end
% At this point, one would usually discard some number of
% the initial samples (the burn-in period). Here, we omit
% this step.
[X,Y] = meshgrid(linspace(-2,2));
figure;
contour(X,Y,p(X,Y)), hold on;
plot(samples(:,1),samples(:,2),'k.')
title(['Single component Gibbs sampler with ',num2str(nsamples),' samples'])
hold off;

figure;
subplot(2,1,1)
plot(samples(:,1));
xlabel('Sample history of x_1');
ylabel('x_1');
hold on;
subplot(2,1,2)
plot(samples(:,2))
xlabel('Sample history of x_2');
ylabel('x_2');

% Note: autocovariances are computed after burn-in is removed!
figure;
ac1 = autocovariance(samples(:,1));
ac2 = autocovariance(samples(:,2));
N_ac = 100;
plot(0:N_ac,ac1(1:N_ac+1),'ro',0:N_ac,ac2(1:N_ac+1),'bo')
pbaspect([1 1 1])
legend('horizontal component','vertical component')
