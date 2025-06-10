function g = generator(n)
% For prime n, find the primitive root modulo n

if ~isprime(n)
    error('n is not a prime');
end

factorlist = unique(factor(n-1));
g = 2; i = 1;
while i <= length(factorlist)
    if powermod(g, (n-1)/factorlist(i), n) == 1
        g = g + 1; i = 0;
    end
    i = i + 1;
end
