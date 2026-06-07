function R = Exp_SO3(f)
R = eye(3) + 1 * skew_SO3(f) + 0.5 * skew_SO3(f)^2;
end