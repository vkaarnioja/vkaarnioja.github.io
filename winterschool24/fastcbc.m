function z = fastcbc(s,n,Gammaratio,gamma)
% Fast CBC construction
% USES: generator.m

m = (n-1)/2;

% Rader permutation
g = generator(n);
perm = ones(1,m);
for j = 1:m-1
    perm(j+1) = mod(perm(j)*g,n);
end
perm = min(perm,n-perm);

% Precompute the FFT of the first column (permuted indices)
bernoulli = @(x) x.^2-x+1/6;
fftomega = fft(bernoulli(mod(perm*perm(end)/n,1)));
z = [];
p = zeros(s,m);
for d = 1:s
    pold = [ones(1,m);p];
    x = gamma(d) * Gammaratio * pold(1:end-1,:);
    if d == 1
        minind = 1;
    else
        tmp = real(ifft(fftomega .* fft(x(fliplr(perm)))));
        [~,minind] = min(tmp);
        minind = perm(minind);
    end
    z = [z;minind];
    omega = bernoulli(mod(minind*(1:m)/n,1));
    for l = 1:d
        p(l,:)=pold(l+1,:)+omega.*pold(l,:)*Gammaratio(l)*gamma(d);
    end
end
