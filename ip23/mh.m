% A simple random walk Metropolis-Hastings algorithm

clear all; close all;
% In general, it is usually preferable to work with
% the logarithm of the target density:
logp = @(x,y) -10*(x.^2-y).^2-(y-1/4).^4;
x = [0,0];
nsamples = 5000;
samples = [];
gamma = .5;
N_accepted = 0;
for iter = 1:nsamples
    step = gamma*randn(1,2);
    y = x + step;
    % Logarithm of the acceptance probability alpha
    logalpha = logp(y(1),y(2)) - logp(x(1),x(2));
    t = rand;
    if logalpha > log(t) % remember to account for the logarithm!
        samples = [samples;y];
        N_accepted = N_accepted + 1;
        x = y;
    else
        samples = [samples;x];
    end
end
% At this point, one would usually discard some number of
% the initial samples (the burn-in period). Here, we omit
% this step.
[X,Y] = meshgrid(linspace(-2,2));
figure;
contour(X,Y,exp(logp(X,Y))), hold on;
plot(samples(:,1),samples(:,2),'k.')
title(['Random walk Metropolis-Hastings with ',num2str(nsamples),' samples, \gamma = ',num2str(gamma),', acceptance ratio ',num2str(N_accepted/nsamples)])
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
title(['$\gamma$ = ',num2str(gamma)],'interpreter','latex','fontsize',14)
