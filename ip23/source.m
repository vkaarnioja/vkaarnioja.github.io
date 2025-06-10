% Source localization example, week 8
clear all; close all;
set(0,'DefaultLineLineWidth',2,'DefaultAxesFontSize',[15],...
    'DefaultAxesFontWeight','bold');
format long

%% Form the posterior density using voltage measurements 
% at both end points x = 0 and x = 1
x_ast = 1/pi; % Fix "ground truth", i.e., particle location 
sigma = .2; % Std for noise
v = 1./abs(x_ast-[0,1]); % Measurements at end points
rng(123); % Reproducible experiments
v = v+sigma*randn(1,2); % Add noise to measurements
x = linspace(0,1); % Discretize the unit interval
% Define the (unnormalized) posterior density
p = @(x) (x >= 0 & x <= 1) .* exp(-1/(2*sigma^2)*((v(1)-1./abs(x-0)).^2+ ...
(v(2)-1./abs(x-1)).^2));
P = cumsum(p(x)); % Form (unnormalized) cumulative distribution function
p_int = 1/(length(x)-1)*(.5*p(x(1))+sum(p(x(2:end-1)))+.5*p(x(end))); % Normalization coefficient (trapezoidal rule)
p_normalized = @(x) p(x) / p_int; % Form normalized posterior density
% Visualize the posterior density and the location of the ground truth
figure;
plot(x,p_normalized(x)), hold on;
plot(x_ast,p_normalized(x_ast),'r*')
legend('posterior density','ground truth');
hold off;

%% Form the posterior density using voltage measurements 
% at only one end points x = 1
x_ast = 1/pi; % Fix "ground truth", i.e., particle location 
sigma = .2; % Std for noise
v = 1./abs(x_ast-[1]); % Measurement at one end point
rng(123); % Reproducible experiments
v = v+sigma*randn(1,1); % Add noise to measurements
x = linspace(0,1); % Discretize the unit interval
% Define the (unnormalized) posterior density
p = @(x) (x >= 0 & x <= 1) .* exp(-1/(2*sigma^2)*(v(1)-1./abs(x-1)).^2);
P = cumsum(p(x)); % Form (unnormalized) cumulative distribution function
p_int = 1/(length(x)-1)*(.5*p(x(1))+sum(p(x(2:end-1)))+.5*p(x(end))); % Normalization coefficient (trapezoidal rule)
p_normalized = @(x) p(x) / p_int; % Form normalized posterior density
% Visualize the posterior density and the location of the ground truth
figure;
plot(x,p_normalized(x)), hold on;
plot(x_ast,p_normalized(x_ast),'r*')
legend('posterior density','ground truth');
hold off;