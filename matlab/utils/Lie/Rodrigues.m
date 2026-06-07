function R = Rodrigues(f)
% t: 3x1
%%
nf = norm(f);
if nf < 1e-100
    R = eye(3);
else
    R = eye(3) + ...
        sin(nf) / nf * skew_SO3(f) + ... 
        (1 - cos(nf)) / (nf^2) * skew_SO3(f)^2;
end

% R = expm(skew_SO3(f));
end