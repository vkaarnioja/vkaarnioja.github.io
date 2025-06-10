%% Backward heat equation example from the sixth lecture.
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

sigma = .01; % Standard deviation of normally distributed noise.
epsilon = sqrt(n-1)*sigma; % Noise level / Morozov discrepancy goal.

rng(123); % Reproducible experiments
U = U_true + sigma*randn(size(U_true));

%% We solve the normal equation using the conjugate gradient method
sol = zeros(n-1,1);   
residual = [];   % Residual of the normal equation
error = [];  % Residual of the original equation
k = 1;
r = A'*U;  % RHS of the normal equation
s = r;
residual(k) = norm(r);
error(k) = norm(U);
figure;
hold on;
h = legend;
while error(k) > epsilon
     hlp = A'*(A*s); % An auxiliary variable.
     alpha = residual(k)^2/(s'*hlp); % The line search in the direction s
     sol = sol + alpha*s; % New CG iterate
     r = r - alpha*hlp; % Update the residual
     residual(k+1) = norm(r); % Update the residual error
     beta =  residual(k+1)^2/residual(k)^2; % Compute the new search direction.
     s = r + beta*s;
     k = k+1;
     error(k) = norm(A*sol - U);
     plot(x,sol,'DisplayName',['k = ',num2str(k-1)]);
end

% Plot the results
plot(x,F_true,'r--','DisplayName','ground truth')
title('Evolution of CG iterates for A^TAF = A^TU');
hold off;

% Check that the Morozov discrepancy goal was achieved.
fprintf('CG reconstruction for A''AF = A''U\n');
disp(['Morozov discrepancy goal: ',num2str(epsilon)]);
disp(['Obtained discrepancy: ',num2str(norm(A*sol-U))]);

figure;
plot(x,sol,'b',x,F_true,'r--')
title(['CG reconstruction for A^TAF = A^TU (k = ',num2str(k-1),' chosen using Morozov)']);
legend('CG reconstruction','ground truth')

figure;
semilogy(error(2:end));
title('Residual ||AF_k-U||')
