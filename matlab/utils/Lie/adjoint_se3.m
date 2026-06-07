function adx = adjoint_se3(x)
w = [x(1);x(2);x(3)];
v = [x(4);x(5);x(6)];
adx = [skew_SO3(w), zeros(3);...
       skew_SO3(v), skew_SO3(w)];
end