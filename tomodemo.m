%% A simple tomography matrix demo
n = 4;
x = -1 + (2/n)*(0:n); % discretization of [-1,1]
% Extending to 2d
[Y,X] = meshgrid(x(1:n)); % ensure that the "labeling" of pixels is preserved
xd = X(:)'; yd = Y(:)';
[Y,X] = meshgrid(x(2:end)); % ditto to the above
xu = X(:)'; yu = Y(:)';
% Note that {(xd(k),yd(k)),(xu(k),yd(k)),(xu(k),yu(k)),(xd(k),yu(k))}
% are the corners of the "kth pixel"!

% In parallel-beam geometry, we (could) have the following setup.
M = 180; % number of illumination angles
K = 150; % number of parallel rays per illumination angle
theta = pi/M * (0:M-1); % equally spaced illumination angles
s = -1 + 2/(K-1) * (0:K-1); % equally spaced "offset" for parallel rays
% Remark: instead of simply letting -1<=s<=1 as is done here, you can scale
% this more generally as -S<=s<=S.

% We want to construct the tomography matrix A in sparse format.
I = []; J = []; val = []; % we need these for A = sparse(I,J,val,M*K,n^2)
for m = 1:M % loop over angles
   cc = cos(theta(m)); ss = sin(theta(m)); % precompute the sine and cosine
   for k = 1:K % loop over rays
       if abs(cc) < eps % horizontal ray
           aux = (yd <= s(k)) .* (s(k) < yu); % which pixels encounter the X-ray?
           I = [I,((m-1)*K+k)*ones(1,sum(aux))]; % labels of X-rays
           J = [J,find(aux)]; % store the "active" pixels
           val = [val,xu(find(aux))-xd(find(aux))];
       elseif abs(ss) < eps % vertical ray
           aux = (xd < s(k)) .* (s(k) <= xu);
           I = [I,((m-1)*K+k)*ones(1,sum(aux))];
           J = [J,find(aux)];
           val = [val,yu(find(aux))-yd(find(aux))];
       elseif cc > 0 % X-ray with downward slope
           tmp = min((xu-s(k)*cc)/ss,(s(k)*ss-yd)/cc)-max((xd-s(k)*cc)/ss,...
               (s(k)*ss-yu)/cc); % cf. pg. 22 of the slides
           aux = tmp > 0; % apply ReLU; this is denoted by ()_+ in the slides 
           I = [I,((m-1)*K+k)*ones(1,sum(aux))];
           J = [J,find(aux)];
           val = [val,tmp(aux)];
       elseif cc < 0 % X-ray with upward slope
           tmp = min((xu-s(k)*cc)/ss,(s(k)*ss-yu)/cc)-max((xd-s(k)*cc)/ss,...
               (s(k)*ss-yd)/cc); % cf. pg. 23 of the slides
           aux = tmp > 0; % apply ReLU; this is denoted by ()_+ in the slides
           I = [I,((m-1)*K+k)*ones(1,sum(aux))];
           J = [J,find(aux)];
           val = [val,tmp(aux)];
       end
   end
end

A = sparse(I,J,val,M*K,n^2); % put everything together
   
%% Sanity check
% Let's consider the simple 4x4 phantom on the Wikipedia page
% https://en.wikipedia.org/wiki/Radon_transform

p = [0,0,0,0;0,1,0,0;0,0,1,0;0,0,0,0];
figure;
imagesc(p), colormap gray

meas = A*p(:);
sino = reshape(meas,K,M); % form the sinogram
figure;
imagesc(sino), colormap gray

% You can also try out this method on the Shepp-Logan phantom
% (function "phantom" in MATLAB).
