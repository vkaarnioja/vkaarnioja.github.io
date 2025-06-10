load sinogram A y M K n
% Projection matrix A
% (noisy) sinogram data y
% nxn grid
% Number of illumination angles M
% Number of parallel X-ray beams K

figure;
imagesc(reshape(y,K,M)), colormap gray

% Construct the discretized (image) gradient operator
block = spdiags([1,-1,1].*ones(n,3),[1-n,0,1],n,n);
LH = [];
for ii = 1:n
    LH = blkdiag(LH,block);
end
LV = spdiags([1,-1,1].*ones(n^2,3),[-n^2+n,0,n],n^2,n^2);
D = [LH;LV];

% Choose CP parameters wisely to ensure convergence
L = svds([A;D],1); % in practice, power method is a good choice here
tau = 1/L;
sigma = 1/L;
theta = 1;

x = zeros(size(A,2),1); q = zeros(size(A,1),1);
z = zeros(2*size(A,2),1); xhat = x;

% Use the Chambolle-Pock algorithm to solve the TV regularized solution
lambda = 15; % a good initial guess for reg. parameter lambda
for ii = 1:10000
   % See pg. 30 of the slides.
    q = (q+sigma*(A*xhat-y))/(1+sigma);
    z = lambda * (z+sigma*D*xhat)./max(lambda,abs(z+sigma*D*xhat));
    xold = x;
    x = max(x-tau*A'*q-tau*D'*z,0);
    xhat = x+theta*(x-xold);
    if mod(ii,1000) == 0
      disp(['iter = ',num2str(ii)]);
    end
end
% (b) Plot the reconstructed x on an nxn grid
figure;
imagesc(reshape(x,n,n)), colormap gray

% (c) Compare with a Tikhonov regularized solution
xtik = [A;lambda*speye(n^2)]\[y;ones(n^2,1)];
figure;
imagesc(reshape(xtik,n,n)), colormap gray
