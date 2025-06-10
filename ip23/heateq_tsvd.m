%% Backward heat equation example from the fourth lecture.
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

% Sanity check
figure;
plot(x,U_true,'k',x,A*F_true,'r--');
legend('Analytic solution','FDM solution')

sigma = .01; % Standard deviation of normally distributed noise.
moro2 = (n-1)*sigma^2; % Noise level squared / Morozov discrepancy goal squared.

rng(123); % Reproducible experiments
% Simulate noisy measurements by adding normally distributed random noise
% with std sigma into the _analytic solution_ to avoid the inverse crime.
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
