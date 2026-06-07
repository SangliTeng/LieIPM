%%
clc;close all;clear all;
import casadi.*

load_system_constant_hiperlab
% dt = 0.01;
% J  = diag([1e-4, 1e-4, 0.28]);
JJ = 1 / 2 * trace(J) * eye(3) - J;

% J * skew_SO3([0; 0; 0.1]);


F  = SX.sym('F', [3, 3]); % SO3xR3 states
dF = SX.sym('dF', [3, 1]);
w  = SX.sym('w', [3, 1]);
dt = SX.sym('dt', [1]);

eq   = J * w * dt - vee_SO3(F * JJ - JJ * F');
J_eq = substitute(eq, F, F * (eye(3) + skew_SO3(dF) ));
J_eq = jacobian(J_eq, dF);
J_eq = substitute(J_eq, dF, dF * 0);
%
% Function('Asym_kkt_mex', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {A_sym});
opts = struct('main', true,...
              'mex',  true);

fun_eq  = Function('omega_eq_mex', {w, F, dt}, {eq} );
fun_Jac = Function('Jomega_eq_mex', {w, F, dt}, {J_eq} );

fun_eq.generate('omega_eq_mex.c',opts);
fun_Jac.generate('Jomega_eq_mex.c',opts);

c_files = dir('*.c');
for k = 1:length(c_files)
    try
        mex(c_files(k).name);
    catch ME
        warning('Failed to compile %s: %s', c_files(k).name, ME.message);
    end
end
%%
output_dir = "gen_casadi";

movefile('*.c', output_dir);
% movefile('*.h', output_dir);            
movefile('*.mex*', output_dir);

%%
dt_num = 0.01
omega = randn(3, 1) * 1;
F0 = eye(3);
%
for k = 1:10
    f_eval = full(omega_eq_mex(omega,  F0, dt_num));
    J_eval = full(Jomega_eq_mex(omega, F0, dt_num));
    F0 = F0 * Rodrigues(-J_eval^-1 * f_eval);
end
%
f_eval
F0

Rodrigues(omega * dt_num)
%
omega
vee_SO3(logm(F0)) / dt_num