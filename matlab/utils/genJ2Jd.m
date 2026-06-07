clear;clc;close all;
%%
syms Jd [3, 3] 'real'
syms J  [3, 3] 'real'

err = trace(Jd) * eye(3) - Jd - J;
err = err(:);

A = jacobian(err, Jd(:));
% A^-1 * (err - A * Jd(:))
J2Jd = A^-1 * J(:);
J2Jd = reshape(J2Jd, [3, 3]);
matlabFunction(J2Jd, vars = {J}, file = "fun_J2Jd.m")