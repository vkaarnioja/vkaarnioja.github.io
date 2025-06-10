clear all; close all;
%% Draw from the Gaussian prior with positivity constraint
p = 100;
alpha = 1;
quantile = @(t) sqrt(2)*alpha*erfinv(t); % inverse CDF
% Use inverse transform sampling: draw "u" from U(0,1)
% and obtain realization by computing quantile(u)
u = rand(p); % draw realization from uniform distribution (0,1)
draw = quantile(u); % evaluate image of u under inverse CDF
figure; imagesc(draw), colormap gray
%figure; surf(draw), colormap gray
title('White noise prior','interpreter','latex','FontSize',[15])

%% Draw from the l^1 prior with positivity constraint
p = 100;
alpha = 1;
quantile = @(t) -1/alpha * log(1-t); % inverse CDF
% Use inverse transform sampling: draw "u" from U(0,1)
% and obtain realization by computing quantile(u)
u = rand(p);
draw = quantile(u); % evaluate image of u under inverse CDF
figure; imagesc(draw), colormap gray
%figure; surf(draw), colormap gray
title('$\ell^1$ prior','interpreter','latex','FontSize',[15])

%% Draw from the Cauchy prior with positivity constraint
p = 100;
alpha = 1;
quantile = @(t) 1/alpha * tan(pi*t/2); % inverse CDF
% Use inverse transform sampling: draw "u" from U(0,1)
% and obtain realization by computing quantile(u)
u = rand(p);
draw = quantile(u); % evaluate image of u under inverse CDF
figure; imagesc(draw), colormap gray
%figure; surf(draw), colormap gray
title('Cauchy prior','interpreter','latex','FontSize',[15])