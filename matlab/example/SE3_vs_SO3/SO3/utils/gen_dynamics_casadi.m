clear all;clc;
% addpath('D:\MatlabPackage\Casadi') 
addpath('C:\ResearchFile\casadi\')
import casadi.*

clc;
load_system_constant;
cons_list = {};
%%
dim_u = 4;
vars = gen_SO3xR3_Variables(cons_list, inertial_list, dim_u);
% input_map = vars.u(1) * zeros(num_srb * 12, 1); %% consider 
% input_map(10) = (vars.u + vars.eu);

R2 = vars.xmat((1:9) + 24, 1); R2 = reshape(R2, [3, 3]);

input_map = [zeros( 6, 1); 
             vars.u(1:3);
             R2 * [0; 0; 1] * vars.u(4)];

[eq_eps, eq_terminal_eps] = gen_SO3xR3_EqualityConstraints(vars, cons_list, inertial_list, g, input_map);
%%
xall = [vars.xmat(1:24, :), vars.xmat(25:end, :)];
xall = xall(:);
opts = struct('main', true,...
              'mex',  true);
[eq, Jac_eq, Hess_eq] = gen_SO3xR3_Jacobians(eq_eps, vars);
vars_fun = {xall, [vars.u; vars.lam], vars.nu, vars.dt};

%%

output_dir = "gen_casadi";

fun_eq = Function('eq_dynamics_mex',vars_fun, {eq});
fun_eq.generate('eq_dynamics_mex.c',opts);

fun_eq_Jac = Function('Jac_dynamics_mex',vars_fun, {Jac_eq});
fun_eq_Jac.generate('Jac_dynamics_mex.c',opts);

fun_eq_Hess = Function('Hess_dynamics_mex',vars_fun, {Hess_eq});
fun_eq_Hess.generate('Hess_dynamics_mex.c',opts);

c_files = dir('*.c');
for k = 1:length(c_files)
    try
        mex(c_files(k).name);
    catch ME
        warning('Failed to compile %s: %s', c_files(k).name, ME.message);
    end
end
%%
movefile('*.c', output_dir);
% movefile('*.h', output_dir);            
movefile('*.mex*', output_dir);