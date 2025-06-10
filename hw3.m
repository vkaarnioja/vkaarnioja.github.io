%% Homework, week 3
% Load projection matrix A, sinogram S and object dimension N.
clear all; close all;
set(0,'DefaultLineLineWidth',2,'DefaultAxesFontSize',[15],...
    'DefaultAxesFontWeight','bold');
% Initialize parameters.
load week3 A S N
noise = 0.01;
epsilon = noise*sqrt(numel(S)); % Morozov discrepancy goal.
y = S(:); % Data vector.

%% Solve the inverse problem 
% We solve the equation Ax = y using Landweber-Fridman iteration. We use
% the Morozov discrepancy principle as the stopping criterion, i.e. we stop
% the algorithm once the residual value reaches epsilon.

% We consider relaxation parameter values beta = [3, 10, 0.01].
beta = [3,10,.01];

% We loop over the different values of beta.
for k = 1:numel(beta)
    
    % Sensibility: the LF scheme converges iff 0<beta<2/lambda(1)^2
    lam = svds(A,1);
    disp(['If beta = ',num2str(beta(k)),', does LF converge: ',...
        num2str(beta(k) < 2/lam^2)]);
    x = zeros(N^2,1);
    iter = 1;
    residual = norm(y);
    while residual(iter) > epsilon
        iter = iter + 1;
        x = x+beta(k)*(A'*(y-A*x)); % The fixed point Landweber-Fridman scheme.
        residual(iter) = norm(A*x-y);
    end

    % Plot the solution.
    figure
    Reco = reshape(x,N,N);
    imagesc(Reco)
    axis 'square'
    colormap 'gray'
    title(sprintf(...
        'Solution using Landweber-Fridman with beta = %.3f\n iterations: %.0f',...
        beta(k),numel(residual)-1));
    
    % Plot the residual.
    figure
    semilogy(0:iter-1,residual)
    title(['Residual using Landweber-Fridman with beta = ',num2str(beta(k))]);
    xlabel('Iteration');
    ylabel('Residual');
end

% Results: the convergence criterion is satisfied for beta = 3 and beta =
% 0.01, but not for beta = 10. We observe that the reconstructions
% corresponding to beta = [3,.01] are mostly equivalent, albeit the first
% relaxation parameter provides a lot faster convergence than the latter.
% The reconstruction using beta = 10 is garbage.
