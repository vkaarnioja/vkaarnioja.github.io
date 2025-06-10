%% Backward heat equation example from the fifth lecture.
close all
set(0,'DefaultLineLineWidth',2,'DefaultAxesFontSize',14,...
    'DefaultAxesFontWeight','bold');

% Initialize the parameters.
T = 0.1;  % Time parameter.
n = 100;  % Number of discretization points.
h = pi/n; % Step size.
x = linspace(h,pi-h,n-1); % Discretization grid.
F_true = double((x >= 1) .* (x <= 2))'; % Ground truth.
Nvec = 1:30; % Use 30 Fourier terms to compute the measurements at time t=T
[X,N] = meshgrid(x,Nvec);
Fcoefs = 2*(cos(N)-cos(2*N))./N/pi; % Fourier sine coefficients
U_true = dot(Fcoefs,exp(-N.^2*T).*sin(N.*X))'; % Analytic solution of the PDE at time t=T

% Construct the system matrix.
% First, use the stencil for the second order spatial derivative.
B = diag(ones(n-2,1),-1) - 2*eye(n-1) + diag(ones(n-2,1),1);
B = 1/h^2 * B;

% Create the forward operator for the temperature distribution at time T.
A = expm(T*B);

sigma = .01; % Standard deviation of normally distributed noise.
moro2 = (n-1)*sigma^2; % Noise level squared / Morozov discrepancy goal squared.

rng(123); % Reproducible experiments
% Simulate noisy measurements by adding normally distributed random noise
% with std sigma into the _analytic solution_ to avoid the inverse crime.
U = A*F_true + sigma*randn(size(U_true));

%% Tikhonov regularization
% Find regularization parameter satisfying the Morozov principle.
tikhonov_solution = @(delta) [A;sqrt(delta)*eye(n-1)]\[U;zeros(n-1,1)];
delta = fminsearch(@(delta) abs(norm(A*tikhonov_solution(delta^2)-U)^2-moro2),.01,optimset('TolX',1e-12));
delta = delta^2; % A lazy method to enforce a positivity constraint for the optimization problem...
    
% Solution with Morozov regularization parameter.
w = tikhonov_solution(delta);

% Check that the Morozov discrepancy goal was achieved.
fprintf('\nTikhonov regularization\n');
disp(['Morozov discrepancy goal: ',num2str(sqrt(moro2))]);
disp(['Obtained discrepancy: ',num2str(norm(U-A*w))]);
disp(['Regularization parameter: ',num2str(delta)]);

% Plot the results
figure;
plot(x,w,'b',x,F_true,'r--');
title(sprintf('Tikhonov reconstruction with delta = %.5e\n chosen using Morozov',delta));
legend('Tikhonov reconstruction','ground truth')

% Investigate what happens when the regularization parameter is too small.
delta = .000001;
w = tikhonov_solution(delta);
figure;
plot(x,w,'b',x,F_true,'r--');
title(sprintf('Tikhonov reconstruction with delta = %.5e\n chosen too small',delta));
    legend('Tikhonov reconstruction','ground truth');
fprintf('\nTikhonov regularization\n');
disp(['Morozov discrepancy goal: ',num2str(sqrt(moro2))]);
disp(['Obtained discrepancy: ',num2str(norm(U-A*w))]);
disp(['Regularization parameter: ',num2str(delta)]);

% Investigate what happens when the regularization parameter is too large.
delta = 1;
w = tikhonov_solution(delta);
figure;
plot(x,w,'b',x,F_true,'r--');
title(sprintf('Tikhonov reconstruction with delta = %.5e\n chosen too large',delta));
    legend('Tikhonov reconstruction','ground truth');
fprintf('\nTikhonov regularization\n');
disp(['Morozov discrepancy goal: ',num2str(sqrt(moro2))]);
disp(['Obtained discrepancy: ',num2str(norm(U-A*w))]);
disp(['Regularization parameter: ',num2str(delta)]);

