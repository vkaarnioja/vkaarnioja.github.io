% Deconvolution example from the lectures of weeks 9 and 10
clear all; close all;
set(0,'DefaultLineLineWidth',2,'DefaultAxesFontSize',14,...
    'DefaultAxesFontWeight','bold');
noiselevel = .1;

% Simulate measurement data using a dense grid
N1 = 150;
s1 = linspace(1/N1/2,1-1/N1/2,N1)';
t1 = s1;
omega = .5;
[T1,S1] = meshgrid(t1,s1);
A = exp(-1/(2*omega^2)*(T1-S1).^2)/N1;
x_true1 = 8*t1.^3 - 16*t1.^2 + 8*t1;
y_01 = A*x_true1;

% Interpolate the data onto a coarser grid and calculate A using the
% coarser grid
N = 120;
s = linspace(1/N/2,1-1/N/2,N)';
t = s;
[T,S] = meshgrid(t,s);
A = exp(-1/(2*omega^2)*(T-S).^2)/N;
x_true = 8*t.^3 - 16*t.^2 + 8*t;
y_0 = interp1(s1,y_01,s);
rng(100); % reproducible experiments
sigma = noiselevel*max(abs(y_0)); % std of the noise
y = y_0 + sigma*randn(size(y_0)); % measurement with noise
Gamma_noise = sigma^2*eye(N); % covariance of the noise
moro = sigma * sqrt(N); % Morozov discrepancy goal
eta_0 = zeros(N,1); % mean of the noise

%% White noise prior
fprintf('Reconstructions with the white noise prior\n');
for gamma = [.2,0.6864,2]
    % Since we have a linear model with additive Gaussian noise and
    % Gaussian prior, the posterior is also Gaussian with explicitly
    % known mean and covariance. Let us draw some samples from both the
    % prior and posterior distributions using the coloring transform.
    % Moreover, we draw the CM estimate of the posterior distribution
    % and draw the 2-sigma credibility envelopes.
    x_0 = zeros(N,1); % prior mean
    Gamma_pr = gamma^2 * eye(N); % prior covariance
    G = A*Gamma_pr*A.' + Gamma_noise; % Kalman gain     
    x_bar = x_0 + Gamma_pr * A.' * inv(G) * (y-A*x_0-eta_0); % posterior mean
    Gamma_post = Gamma_pr - Gamma_pr * A.' * inv(G) * A * Gamma_pr; % posterior covariance
    Gamma_post = (Gamma_post+Gamma_post')/2;
    
    % Remark: The "inv" command is used rather liberally in what follows.
    % As a rule of thumb, it is usually better to use matrix decompositions
    % such as LU or Cholesky decomposition or numerical linear system solvers.

    % Draw some samples from the prior
    R_pr = chol(inv(Gamma_pr));
    samples_pr = R_pr \ randn(N,10) + x_0; % coloring transform
    %samples_pr = mvnrnd(x_0,Gamma_pr,10); % sampling a Gaussian density using mvnrnd
    figure;
    plot(t,samples_pr','LineWidth',1);
    title(['Samples drawn from the white noise prior, \gamma = ',num2str(gamma)]);

    % Draw some samples from the posterior
    R_post = chol(inv(Gamma_post));
    samples_post = R_post \ randn(N,10) + x_bar; % coloring transform
    %samples_post = mvnrnd(x_bar,Gamma_post,10); % sampling a Gaussian density using mvnrnd
    
    % We can marginalize the posterior w.r.t. each component in order
    % to obtain the 2-sigma credibility envelopes.
    variances = diag(Gamma_post);
    figure;
    plot(t,x_true,'r',t,x_bar,'b',t,x_bar+2*sqrt(variances(:)),'b',t,x_bar-2*sqrt(variances(:)),'b','LineWidth',3), hold on;
    plot(t,samples_post','LineWidth',1)
    legend('ground truth','posterior mean \pm 2\sigma')
    title(['Samples drawn from the posterior with white noise prior, \gamma = ',num2str(gamma)])
    hold off;
    
    disp(['gamma = ',num2str(gamma)]);
    disp(['Morozov discrepancy goal: ',num2str(moro)]);
    disp(['Obtained discrepancy: ',num2str(norm(A*x_bar-y))]);
end

%% Smoothness prior
L = -diag(ones(N-1,1),-1) + 2*eye(N) - diag(ones(N-1,1),1);
fprintf('\nReconstructions with the smoothness prior\n');
for gamma = [0.001,0.0064,0.02]
    % Since we have a linear model with additive Gaussian noise and
    % Gaussian prior, the posterior is also Gaussian with explicitly
    % known mean and covariance. Let us draw some samples from both the
    % prior and posterior distributions using the coloring transform.
    % Moreover, we draw the CM estimate of the posterior distribution
    % and draw the 2-sigma credibility envelopes.
    x_0 = zeros(N,1); % prior mean
    Gamma_pr = gamma^2 * inv(L'*L); % prior covariance
    Gamma_pr = (Gamma_pr+Gamma_pr')/2;
    G = A*Gamma_pr*A.' + Gamma_noise; % Kalman gain  
    x_bar = x_0 + Gamma_pr * A.' * inv(G) * (y-A*x_0-eta_0); % posterior mean      
    Gamma_post = Gamma_pr - Gamma_pr * A.' * inv(G) * A * Gamma_pr; % posterior covariance
    Gamma_post = (Gamma_post+Gamma_post')/2;
    
    % Remark: The "inv" command is used rather liberally in what follows.
    % As a rule of thumb, it is usually better to use matrix decompositions
    % such as LU or Cholesky decomposition or numerical linear system solvers.

    % Draw some samples from the prior
    R_pr = chol(inv(Gamma_pr));
    samples_pr = R_pr \ randn(N,10) + x_0; % coloring transform
    %samples_pr = mvnrnd(zeros(N,1),Gamma_pr,10); % sampling a Gaussian density using mvnrnd
    figure;
    plot(t,samples_pr','LineWidth',1);
    title(['Samples drawn from the smoothness prior, \gamma = ',num2str(gamma)]);

    % Draw some samples from the posterior
    R_post = chol(inv(Gamma_post));
    samples_post = R_post \ randn(N,10) + x_bar; % coloring transform
    %samples_post = mvnrnd(x_bar,Gamma_post,10); % sampling a Gaussian density using mvnrnd
    
    % We can marginalize the posterior w.r.t. each component in order
    % to obtain the 2-sigma credibility envelopes.
    variances = diag(Gamma_post);
    figure;
    plot(t,x_true,'r',t,x_bar,'b',t,x_bar+2*sqrt(variances(:)),'b',t,x_bar-2*sqrt(variances(:)),'b','LineWidth',3), hold on;
    plot(t,samples_post','LineWidth',1)
    legend('ground truth','posterior mean \pm 2\sigma')
    title(['Samples drawn from the posterior with smoothness prior, \gamma = ',num2str(gamma)])
    hold off;   
    
    disp(['gamma = ',num2str(gamma)]);
    disp(['Morozov discrepancy goal: ',num2str(moro)]);
    disp(['Obtained discrepancy: ',num2str(norm(A*x_bar-y))]);
end

