% Let us try using the TSVD method to solve an X-ray tomography problem.
% We use the FIPS open data set https://zenodo.org/record/1254210

load DataFull_128x15.mat % sparse angle tomography data
%load DataLimited_128x15.mat % limited angle tomography data

% Solving the linear system _without_ using regularization.
res_naive = A\m(:);
figure;
imagesc(reshape(res_naive,128,128)), colormap gray
drawnow;
% The problem is very ill-conditioned, so the the presence of small
% measurement noise ruins the reconstruction.

%% We form the truncated SVD of the system matrix corresponding to
% k singular values and solve the spectrally truncated equation.
for k = [1,10,100,1000]
    [u,s,v] = svds(A,k);
    Sinv = diag(1./diag(s));
    res = v*(Sinv*(u'*m(:)));
    figure;
    imagesc(reshape(res,128,128)), colormap gray
    drawnow;
end
% The reconstruction corresponding to 1000 largest singular values looks
% reasonable. While the TSVD method is simple to implement, the implementation 
% can be extremely slow if the (sparse) system matrix is very large in size.