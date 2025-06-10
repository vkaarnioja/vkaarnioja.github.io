% Total variation regularization for X-ray tomography

% Import projection matrix A and the (noisy) sinogram m
load sino A S N
y = S(:); % vectorize the sinogram

% Visualize the sinogram data
figure;
imagesc(S), colormap gray
drawnow;

% Construct the discretized (image) gradient operator
block = spdiags([1,-1,1].*ones(N,3),[1-N,0,1],N,N);
LH = [];
for ii = 1:N
    LH = blkdiag(LH,block);
end
LV = spdiags([1,-1,1].*ones(N^2,3),[-N^2+N,0,N],N^2,N^2);
D = [LH;LV];

% Choose CP parameters wisely to ensure convergence
L = svds([A;D],1,'largest'); 
tau = 1/L;
sigma = 1/L;
theta = 1;
x = zeros(size(A,2),1); q = zeros(size(A,1),1);
z = zeros(size(D,1),1); xhat = x;

% Use the Chambolle-Pock algorithm to solve the TV regularized solution
lambda = .01; % educated guess for the regularization parameter
for ii = 1:1000
    % See pg. 30 of the slides.
    q = (q+sigma*(A*xhat-y))/(1+sigma);
    z = lambda * (z+sigma*D*xhat)./max(lambda,abs(z+sigma*D*xhat));
    xold = x;
    x = max(x-tau*A'*q-tau*D'*z,0);
    xhat = x+theta*(x-xold);
end

% Plot the reconstructed x on an NxN grid
figure;
imagesc(reshape(x,N,N)), axis square, colormap gray
drawnow;

%% Compare with a Tikhonov regularized solution
xtik = [A;sqrt(lambda)*speye(N^2)]\[y;zeros(N^2,1)];
figure;
imagesc(reshape(xtik,N,N)), axis square, colormap gray
drawnow;