function R = Exp_SO3_sym(f)
% t: 3x1
%%
nf = norm(f) + 1e-8;
R = eye(3) + ...
    sin(nf) / nf * skew_SO3(f) + ...
    (1 - cos(nf)) / (nf^2) * skew_SO3(f)^2;
end
