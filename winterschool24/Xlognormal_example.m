% PDE uncertainty quantification for the "lognormal model" using QMC
% USES: Partial Differential Equation Toolbox, Parallel Computing Toolbox,
%       fastcbc.m, generator.m
clear all; close all;

% Create a FE mesh
order = 'linear'; % use first-order FEM
h = .03; % mesh size
model = createpde();
g = decsg([3;4;0;1;1;0;0;0;1;1]); % define the geometry
geometryFromEdges(model,g);
generateMesh(model,'GeometricOrder',order,'Hmax',h);
[p,~,t] = meshToPet(model.Mesh); % store the FE mesh

%Visualize the mesh
%figure;
%pdemesh(model);
%drawnow;

% Precompute the mass matrix and loading term
f = @(location,state) location.x; % source term f(x)
specifyCoefficients(model,'m',1,'d',0,'c',1,'a',0,'f',f); % coerce MATLAB to form the mass matrix
applyBoundaryCondition(model,'dirichlet','Edge',1:4,'u',0); % enforce Dirichlet zero BCs
FEM = assembleFEMatrices(model,'nullspace'); % assemble FE matrices
mass = FEM.M; % mass matrix
loading = FEM.Fc; % loading term
[interior,~] = find(FEM.B); % find indices corresponding to interior nodes
ndof = length(mass); % number of degrees of freedom
ncoord = length(p); % total number of FE nodes (including boundary nodes)
clear model;

% Problem set-up
s = 100; % stochastic dimension
decay = 2; % decay of the input random field
onevec = ones(ndof,1); % initialize a vector of ones

% Initialize the parallel pool
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

% Weight structure
b = (1:s).^(-decay);
delta = .05;
lambda = 1/(2-2*delta);
alpha = 0.5*(b+sqrt(b.^2+1-1/(2*lambda)));
rho = 2*(sqrt(2*pi)*exp(alpha.^2*4*lambda/(2*lambda-1))./...
    (pi^(2-(2*lambda-1)/2/lambda)*(1-(2*lambda-1)/4/lambda)*(2*lambda-1)/4/lambda)).^lambda*zeta(lambda+.5);
Gammaratio = (1:s).^(2/(1+lambda));
gamma = (b./(2*log(2)*exp(b.^2/2).*normcdf(b).*sqrt((alpha-b).*rho))).^(2/(1+lambda));

R = 8; % number of random shifts
nlist = [17,31,67,127,257,503,1009,2003,4001,8009,16007,32003,64007];
rms = [];
for n = nlist
    disp(['n = ',num2str(n)]);
    z = fastcbc(s,n,Gammaratio,gamma); % find generating vector
    results = [];
    for r = 1:R
        shift = rand(s,1); % random shift
        result = 0;
        parfor ii = 1:n
            y = norminv(mod(ii*z/n+shift,1)); % randomly shifted lattice point
            model = createpde();
            geometryFromMesh(model,p,t(1:end-1,:)); % use the precomputed FE mesh
            % Parameterization of the diffusion coefficient
            a = @(location,state) exp(y'*([1:s]'.^-decay.*sin([1:s]'*pi*location.x).*sin([1:s]'*pi*location.y))); % coefficient function a(x,y)
            specifyCoefficients(model,'m',0,'d',0,'c',a,'a',0,'f',f);
            applyBoundaryCondition(model,'dirichlet','Edge',1:4,'u',0); % enforce Dirichlet zero BCs
            FEM = assembleFEMatrices(model,'K'); % assemble the stiffness matrix
            stiffness = FEM.K(interior,interior); % we only need the part corresponding to interior nodes
            sol = stiffness\loading; % solve the PDE
            result = result + onevec'*mass*sol; % sum up the results
        end
        results = [results;result/n]; % compute the QMC average
    end
    qmcavg = mean(results); % compute the estimator
    rmserror = norm(qmcavg-results)/sqrt(R*(R-1)); % R.M.S. error estimate
    rms = [rms;rmserror];
end

% Least squares fit
nadjusted = R*nlist;
lsq = [ones(size(nadjusted,2),1),log(nadjusted')]\log(rms);
lsq(1) = exp(lsq(1));

% Visualize the results
figure;
loglog(nadjusted,rms,'r.','MarkerSize',18), hold on
loglog(nadjusted,lsq(1)*nadjusted.^lsq(2),'b','LineWidth',2), hold off
set(gca,'FontSize',14,'FontWeight','bold')
legend('QMC error',['slope: ',num2str(lsq(2))]);
title(['QMC error (s = ',num2str(s),')'])
xlabel('n');
ylabel('error')
