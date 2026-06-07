function [val, Jac, Hess] = gen_SO3xR3_Jacobians(eq, vars)
import casadi.*
% lam = vars.lam;
% xmat = vars.xmat;
% emat = vars.emat;
% dt = vars.dt;
emat_temporal = vars.emat_temporal;
elam = vars.elam;
eu = vars.eu;
nu = vars.nu;

vars = [emat_temporal(:); eu(:); elam(:)];

opts = struct();
%%
val = Function('val', {vars}, {eq}, opts); 
val = val(vars * 0);

Jac = jacobian(eq, vars);
Jac = Function('Jac', {vars}, {Jac},opts);
Jac = Jac(vars * 0);

Hess = 0;
for k = 1:length(eq)
    Hess = Hess + hessian(eq(k), vars) * nu(k);
end

Hess = Function('Hess', {vars}, {Hess},opts);
Hess = Hess(vars * 0);

end