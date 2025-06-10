function gamma2 = autocovariance(f)
N = length(f);
f_c = f-mean(f);
gamma2_0 = mean(f_c.^2);
for k = N-1:-1:1
   jj = 1:N-k;
   gamma2(k) = 1/(gamma2_0*(N-k)) * sum(f_c(jj).*f_c(jj+k));
end