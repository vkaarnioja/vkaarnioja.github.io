% Isospectral drums
%
% Two drums produce the same sound if they have the same Dirichlet
% eigenvalues -div(grad(u)) = lam*u, u = 0 on the boundary.
%
% Written by Vesa Kaarnioja, 2018

%% First polygon
h = .1;
[x,y] = meshgrid(0:h:3);
mesh = [x(:),y(:)];

xv = [2,3,3,1,1,0,1,2]; % x-coordinates of the polygon's vertices
yv = [0,1,2,2,3,2,1,1]; % y-coordinates of the polygon's vertices

[in,on] = inpolygon(mesh(:,1),mesh(:,2),xv,yv); % see: "doc inpolygon"
non = logical(in-on); % remove from interior nodes the nodes on the boundary

figure;
plot(mesh(non,1),mesh(non,2),'k+'), axis equal

mesh = mesh(non,:); % Now mesh contains the interior nodes
n = size(mesh,1); % system size

% Let's create the system matrix!
A = sparse(n,n); % nxn zero matrix
for i=1:n
    for j=1:n
        if (norm(mesh(i,:)-mesh(j,:)) < 1.1*h)
            A(i,j) = 1;
        end
    end
end

% Remember that the entries on the diagonal are -4
A = A-5*speye(n); % Ed: speye keeps A in sparse format!
A = A/h^2;

%% The second polygon
[x,y] = meshgrid(0:h:3);
mesh = [x(:),y(:)];

xv = [2,2,3,2,1,1,0,0]; % x-coordinates of the polygon's vertices
yv = [0,1,1,2,2,3,3,2]; % y-coordinates of the polygon's vertices

[in,on] = inpolygon(mesh(:,1),mesh(:,2),xv,yv);
non = logical(in-on); % remove from interior nodes the nodes on the boundary

figure;
plot(mesh(non,1),mesh(non,2),'k+'), axis equal

mesh = mesh(non,:); % Now mesh contains the interior nodes
n = size(mesh,1); % system size

% Let's create the system matrix!
B = sparse(n,n);
for i=1:n
    for j=1:n
        if (norm(mesh(i,:)-mesh(j,:)) < 1.1*h)
            B(i,j) = 1;
        end
    end
end

% Remember that the entries on the diagonal are -4.
B = B-5*speye(n); % Ed: speye keeps B in sparse format!
B = B/h^2;

norm(eig(A)-eig(B)) % should be close to 0