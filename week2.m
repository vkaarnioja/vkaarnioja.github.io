%% Backward heat equation example from the second lecture.
close all
set(0,'DefaultLineLineWidth',2,'DefaultAxesFontSize',14,...
    'DefaultAxesFontWeight','bold');

% Initialize the parameters.
T = 0.1; % Time parameter.
n = 100; % Number of discretization points.
h = pi/n;% Step size.
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

%Sanity check
figure;
plot(x,U_true,'k',x,A*F_true,'r--');
legend('Analytic solution','FDM solution')

sigma = .01; % Standard deviation of normally distributed noise.
moro2 = (n-1)*sigma^2; % Noise level squared / Morozov discrepancy goal squared.

rng(123); % Reproducible experiments
U = U_true + sigma*randn(size(U_true));

% Singular values of system matrix A
[u,s,v] = svd(A);
s = diag(s);
figure;
semilogy(s);
title('Singular values of the system matrix')

% Minimum norm solution for noisy data (not regularized)
U_bad = pinv(A)*U;
figure;
plot(x,U_bad,'r');
title('Minimum norm solution (not regularized)');

% TSVD solution with Morozov principle
ind = 5;
Sp1 = diag([1./s(1:ind);zeros(n-ind-1,1)]);
Ap1 = v*Sp1*u';
res = Ap1*U;
% Morozov discrepancy principle is approximately satisfied
fprintf('TSVD regularization\n');
disp(['Morozov discrepancy goal: ',num2str(sqrt(moro2))]);
disp(['Obtained discrepancy: ',num2str(norm(A*res-U))]);
disp(['Spectral cut-off: k = ',num2str(ind)]);
figure;
plot(x,res,'b',x,F_true,'r--');
title(['TSVD reconstruction (k = ',num2str(ind),' chosen using Morozov)']);
legend('TSVD reconstruction','ground truth');

% TSVD solution with too small spectral cut-off
ind = 2;
Sp1 = diag([1./s(1:ind);zeros(n-ind-1,1)]);
Ap1 = v*Sp1*u';
res = Ap1*U;
fprintf('\nTSVD regularization\n');
disp(['Morozov discrepancy goal: ',num2str(sqrt(moro2))]);
disp(['Obtained discrepancy: ',num2str(norm(A*res-U))]);
disp(['Spectral cut-off: k = ',num2str(ind)]);
figure;
plot(x,res,'b',x,F_true,'r--');
title(['TSVD reconstruction (k = ',num2str(ind),' too small)']);
legend('TSVD reconstruction','ground truth');

% TSVD solution with too large spectral cut-off
ind = 8;
Sp1 = diag([1./s(1:ind);zeros(n-ind-1,1)]);
Ap1 = v*Sp1*u';
res = Ap1*U;
fprintf('\nTSVD regularization\n');
disp(['Morozov discrepancy goal: ',num2str(sqrt(moro2))]);
disp(['Obtained discrepancy: ',num2str(norm(A*res-U))]);
disp(['Spectral cut-off: k = ',num2str(ind)]);
figure;
plot(x,res,'b',x,F_true,'r--')
title(['TSVD reconstruction (k = ',num2str(ind),' too large)']);
legend('TSVD reconstruction','ground truth')

% Use Newton's method to solve for the regularization parameter satisfying
% the Morozov discrepancy principle.
prec = 10^(-10); % Stopping criterion for Newton's method.

% Initializations
y_aug = [U;zeros(n-1,1)]; % Stacked form of the measurement data.
delta = 0.01; % Initial guess for the regularization parameter.
discr2 = 1; % The discrepancy.
    
% Use Newton's method to find delta such that norm(U-A*w)^2 == moro^2.
while discr2 > prec
    K = [A;sqrt(delta)*eye(n-1)]; % Stacked form of the Tikhonov operator.
    w = K\y_aug; % Tikhonov regularized solution.
    err = norm(U - A*w)^2; % The discrepancy.
    discr2 = abs(err-moro2);

    % Derivative of discrepancy w.r.t. delta.
    d_err = 2*delta*w'*(inv(A'*A + delta*eye(n-1))*w);

    % Newton step to update the regularization parameter.
    delta = delta + (moro2 - err)/d_err;
end

% Solution with Morozov regularization parameter.
K = [A;sqrt(delta)*eye(n-1)];
w = K\y_aug;

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

delta = .000001;
K = [A;sqrt(delta)*eye(n-1)];
w = K\y_aug;
figure;
plot(x,w,'b',x,F_true,'r--');
title(sprintf('Tikhonov reconstruction from with delta = %.5e\n chosen too small',delta));
    legend('Tikhonov reconstruction','ground truth');
fprintf('\nTikhonov regularization\n');
disp(['Morozov discrepancy goal: ',num2str(sqrt(moro2))]);
disp(['Obtained discrepancy: ',num2str(norm(U-A*w))]);
disp(['Regularization parameter: ',num2str(delta)]);

delta = 1;
K = [A;sqrt(delta)*eye(n-1)];
w = K\y_aug;
figure;
plot(x,w,'b',x,F_true,'r--');
title(sprintf('Tikhonov reconstruction with delta = %.5e\n chosen too large',delta));
    legend('Tikhonov reconstruction','ground truth');
fprintf('\nTikhonov regularization\n');
disp(['Morozov discrepancy goal: ',num2str(sqrt(moro2))]);
disp(['Obtained discrepancy: ',num2str(norm(U-A*w))]);
disp(['Regularization parameter: ',num2str(delta)]);
