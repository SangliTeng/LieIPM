clear all;clc;
% TSL: replace with your own path and version
addpath('C:\ResearchFile\casadi-windows-matlabR2016a-v3.5.5\')
import casadi.*
%%
warning off
clc;
load_system_constant_122_new; % define the mass and inertia here
cons_list = {}; % for the constrained systems, you have to define the dimensions and a function to generate the constraints.
%% Dynamics of a single step 
dim_u = 4;
vars = gen_SO3xR3_Variables(cons_list, inertial_list, dim_u); % define all the variables and the perturbations

R2 = vars.xmat((1:9) + 24, 1); R2 = reshape(R2, [3, 3]);

% B1 = 0.041010498687664041987;
% B2 = 0.0051187551187551187549;
% B3 = 1;
% 
% B = [-B1, -B1,  B1,  B1;
%      -B1,  B1,  B1, -B1;
%      -B2,  B2, -B2,  B2;
%       B3,  B3,  B3,  B3];

tau_f = B * vars.u;


input_map = [zeros( 6, 1); 
             tau_f(1:3);
             R2 * [0; 0; 1] * tau_f(end)]; % define the input map

% to test
% [eq_eps, eq_terminal_eps] = gen_SO3xR3_EqualityConstraints_trace(vars, cons_list, inertial_list, g, input_map);

% in the RSS 2025 paper
[eq_eps, eq_terminal_eps] = gen_SO3xR3_EqualityConstraints(vars, cons_list, inertial_list, g, input_map); 

%% State for all steps 
% Ns = 20;

x  = SX.sym('x',  [24 * size(vars, 2), Ns + 1]); % SO3xR3 states
u  = SX.sym('u',  [dim_u, Ns]); % control inputs 
lam = SX.sym('lam', [12 * size(vars.lam, 2), Ns]); % constrained forces, should be merged to control

y = SX.sym('y', [size(vars.nu, 1), Ns + 1] ); % multiplier for all the equality constraints

% the following starting with e are the perturbations
ex  = SX.sym('ex',  [12 * size(vars.emat, 2), Ns + 1]);
eu = SX.sym('eu', [dim_u * size(vars.eu, 2), Ns]);
elam = SX.sym('elam', [12 * size(vars.elam, 2) * 0, Ns]);
ey = SX.sym('ey', size(y));

dt = SX.sym('dt',  [Ns]);

Lagrangian = 0;
%% The retraction map. 
% X \times Rn \times Rm
x_new = [];
for k = 1:Ns + 1
    Rk = x(1:9, k); Rk = reshape(Rk, [3, 3]);
    pk = x(10:12, k);
    Fk = x(13:21, k); Fk = reshape(Fk, [3, 3]);
    vk = x(22:24, k);

    Rk_new = Rk * Exp_SO3_Rodriguez(ex(1:3, k));
    pk_new = pk + ex(4:6, k);
    Fk_new = Fk * Exp_SO3_Rodriguez(ex(7:9, k));
    vk_new = vk + ex(10:12, k);

    xk_new = [Rk_new(:);
              pk_new(:);
              Fk_new(:);
              vk_new(:)];

    x_new = [x_new, xk_new];
end

y_new = y + ey;
u_new = u + eu;

%% The following code generate the Lagrangian:
% L = c(x) + \sum yi * hi(x) + \sum zi * g(x)
%% Cost in the Lagrangian
% c(x)
[SO3xR3_cost, vars_cost] = gen_SO3xR3_trace_cost(); % stage cost
% [SO3xR3_cost, vars_cost] = gen_SO3xR3_flat_cost(); % stage cost
% [SO3xR3_cost, vars_cost] = gen_SO3xR3_trace_huber_loss(); % stage cost
% [SO3xR3_cost, vars_cost] = gen_SO3xR3_geodesic_huber_loss(); % stage cost

% terminal cost
QRt = SX.sym('QRt', 3, 3);
QFt = SX.sym('QFt', 3, 3);
Qpt = SX.sym('Qpt', 3, 3);
Qvt = SX.sym('Qvt', 3, 3);

xd  = SX.sym('xd',  [24 * size(vars, 2), Ns + 1]);
ud  = SX.sym('ud',  [dim_u, Ns]);

Rcontrol = SX.sym('R', [dim_u, dim_u]);
for k = 1:Ns+1
    ck = substitute(SO3xR3_cost, vars_cost.x,  x(:, k));
    ck = substitute(ck, vars_cost.xd, xd(:, k));
    ck = substitute(ck, vars_cost.e,   ex(:, k));
    
    if k == Ns + 1
        ck = substitute(ck, vars_cost.QR,   QRt);
        ck = substitute(ck, vars_cost.Qp,   Qpt);
        ck = substitute(ck, vars_cost.QF,   QFt);
        ck = substitute(ck, vars_cost.Qv,   Qvt);
    else
        du = (u(:, k) + eu(:, k) - ud(:, k));
        ck = ck + 1 / 2 * du' * Rcontrol * du; % // quadratic
        % ck = ck + sqrt( du' * Rcontrol * du  + 1e-2);
        
        % a small regularizer for symmetry... 
        % ck = ck + ((du(1) - du(3))^2 + (du(2) - du(4))^2 ) * 1e-2;
    end

    Lagrangian = Lagrangian + ck;
end

Cost = Lagrangian;
%% Dynamics constraints in the Lagrangian
% \sum yi * hi(x)
eq_all = [];
for k = 1:Ns
    eq_k   = substitute(eq_eps, vars.emat, [ex(:, k); ex(:, k+1)] );
    eq_k   = substitute(eq_k,   vars.eu,    eu(:, k) );
    eq_k   = substitute(eq_k,   vars.elam,    elam(:, k) );

    eq_k   = substitute(eq_k,   vars.xmat,  [x(:, k); x(:, k+1)] );
    eq_k   = substitute(eq_k,   vars.u,      u(:, k) );
    eq_k   = substitute(eq_k,   vars.dt,     dt(k) );
    % eq_k   = substitute(eq_k,   vars.lam,    lam(:, k) );
    eq_all = [eq_all; eq_k];

    Lagrangian = Lagrangian + sum(eq_k .* y(:, k+1));
end
%% Inequality constraints in the Lagrangian
% \sum zi * g(x)
ieq_all = [];
dim_ieq = 8;
z  = SX.sym('z', [dim_ieq, Ns]);
s  = SX.sym('s', size(z));
ez = SX.sym('ez', [dim_ieq, Ns]);
es = SX.sym('es', size(z));

for k = 1:Ns
    ieq_k = [ (u(:, k) + eu(:, k)) - 3; % 4;
             -(u(:, k) + eu(:, k)) - 0;
             % -(x(12, k+1)+ex(6, k+1)) - 1; %% height constriants
             % -(x(12)+ex(3)) + (x(10)+ex(1))
             ]; % [4;4;4;0]];
              % (u(4, k) + eu(4, k)) - 4;
             % -(u(4, k) + eu(4, k))];
    % ieq_k = ieq_k([1, 5, 2, 6, 3, 7, 4, 8]);
    % ieq_k = ieq_k;
    ieq_all = [ieq_all; ieq_k + s(:, k) ];
    Lagrangian = Lagrangian + sum(ieq_k .* z(:, k));
end

z_new = z + ez;
s_new = s + es;


%% Initial constraints
% \sum yi * (x0 - xinit)_i
% R0 = SX.sym('R0', [3, 3]);
% p0 = SX.sym('p0', [3, 3]);
% F0 = SX.sym('F0', [3, 3]);
% v0 = SX.sym('v0', [3, 3]);
% import casadi.*

x0 = SX.sym('x0', [24, 1]);

eq_0 = [Log_SO3(reshape(x0(1:9), [3, 3])' * reshape(x(1:9), [3, 3])) + ex(1:3)';
        x(10:12)' + ex(4:6)' - x0(10:12);
        Log_SO3(reshape(x0(13:21), [3, 3])' * reshape(x(13:21), [3, 3])) + ex(7:9)';
        x(22:24)' + ex(10:12)' - x0(22:24)];

eq_all = [eq_0; eq_all];

Lagrangian = Lagrangian + sum(eq_0 .* y(:, 1));

vars_kkt = [ex(:); eu(:); y(:); z(:); s(:)];

mu = SX.sym('mu', 1); % homotopy parameters

%% KKT vector field and its Jacobians
b = [jacobian(Lagrangian, [ex(:); eu(:)] )';
                    eq_all;
                    ieq_all;
                    s(:) .* z(:) - mu];

A = jacobian(b, vars_kkt);

% symmetric submatrix of A for symmetric linear solver that might be more
% efficnet. According to IPOPT paper
% b = jacobian(Lagrangian, vars_kkt);
% A = hessian(Lagrangian, vars_kkt);

f  = substitute(Cost, [ex(:); eu(:)], [ex(:); eu(:)] * 0); % cost 
df = jacobian(Cost, [ex(:); eu(:)]);
df = substitute(df, [ex(:); eu(:)], [ex(:); eu(:)] * 0); % Jacobian of the cost
b  = substitute(b, [ex(:); eu(:)], [ex(:); eu(:)] * 0); % vector field
A  = substitute(A, [ex(:); eu(:)], [ex(:); eu(:)] * 0); % Jacobians
len_sym = length(ex(:)) + length(eu(:)) + length(y(:)) + length(z(:));
% A(1:len_sym, 1:len_sym) = (A(1:len_sym, 1:len_sym)' + A(1:len_sym, 1:len_sym)) / 2;

% A(1000, 1000)
A = A + speye(size(A, 1)) * 1e-2;
A = A - speye(size(A, 1)) * 1e-2;
% A(1000, 1000)

h = substitute(eq_all, [ex(:); eu(:)], [ex(:); eu(:)] * 0); % equality constraints 
g = substitute(ieq_all, [ex(:); eu(:)], [ex(:); eu(:)] * 0); % inequality constraints 
g = substitute(g, s(:), s(:) * 0);

A_sym = A(1:len_sym, 1:len_sym);
A_sym(end-length(s(:))+1:end, end-length(s(:))+1:end) = - diag( s(:) ./ z(:));
A_sym = triu(A_sym);
% A_sym(end-length(s(:))+1:end, end-length(s(:))+1:end) = A_sym(end-length(s(:))+1:end, end-length(s(:))+1:end) - 1e-2 *speye(length(s(:))); % enforce inertia correction...
% A_sym(1:length([ex(:); eu(:)]), 1:length([ex(:); eu(:)])) = A_sym(1:length([ex(:); eu(:)]), 1:length([ex(:); eu(:)])) - 1 *speye(length([ex(:); eu(:)]));

% A(1:len_sym, 1:len_sym) = (A(1:len_sym, 1:len_sym) + A(1:len_sym, 1:len_sym)') / 2;
% A(1:len_sym, 1:len_sym) = triu(A(1:len_sym, 1:len_sym));
%%

Qstage = [vars_cost.QR(:);
          vars_cost.Qp(:);
          vars_cost.QF(:); 
          vars_cost.Qv(:)];
Qterminal = [QRt(:);
             Qpt(:);
             QFt(:); 
             Qvt(:)];

% fun_b = Function('b', {x, x0, Qstage, Qterminal, xd, ud}, {b})
% fun_b = Function('b', {x, x0, Qstage, Qterminal, xd, ud}, {b})

opts = struct('main', true,...
              'mex',  true);

xu = [x(:); u(:)];

%% generate the util functions

fun_b = Function('b_kkt_mex', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {b});
fun_b.generate('b_kkt_mex.c',opts);

fun_A = Function('A_kkt_mex', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {A});
fun_A.generate('A_kkt_mex.c',opts);

fun_Asym = Function('Asym_kkt_mex', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {A_sym});
fun_Asym.generate('Asym_kkt_mex.c',opts);
% 
fun_f = Function('f_mex', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {f});
fun_f.generate('f_mex.c',opts);

fun_df = Function('df_mex', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {df});
fun_df.generate('df_mex.c',opts);

fun_hg = Function('hg_mex', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {h, g});
fun_hg.generate('hg_mex.c',opts);

xu_new = [x_new(:); u_new(:)]; % for the solver, the primal variables are both the states and the input. 

fun_retract = Function('Retraction_xyz',...
                       {xu, y(:), z(:), s(:), [ex(:); eu(:); ey(:); ez(:); es(:) ]},...
                       { xu_new, y_new(:), z_new(:), s_new(:)});
fun_retract.generate('Retraction_xyz_mex.c',opts);

%%
c_files = dir('*.c');
parfor k = 1:length(c_files)
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
%% the following are for the cpp applications outside of MATLAB
import casadi.*
output_dir = "gen_casadi_cpp";
opts = struct('with_header', true) % , 'precision', 'single');

xu = [x(:); u(:)];

fun_b = Function('b_kkt', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {b});
fun_b.generate('b_kkt.cpp',opts);

fun_A = Function('A_kkt', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {A});
fun_A.generate('A_kkt.cpp',opts);

fun_Asym = Function('Asym_kkt', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {A_sym});
fun_Asym.generate('Asym_kkt.cpp',opts);

fun_f = Function('f', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {f});
fun_f.generate('f.cpp',opts);

fun_df = Function('df', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {df});
fun_df.generate('df.cpp',opts);

fun_hg = Function('hg', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)}, {h, g});
fun_hg.generate('hg.cpp',opts);


% fun_utils = Function('utils', {x, u, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd, ud, dt}, {A, b, f, df});
% fun_utils.generate('utils_mex.c',opts);

xu_new = [x_new(:); u_new(:)];

fun_retract = Function('Retraction_xyz',...
                       {xu, y(:), z(:), s(:), [ex(:); eu(:); ey(:); ez(:); es(:) ]},...
                       { xu_new, y_new(:), z_new(:), s_new(:)});
fun_retract.generate('Retraction_xyz.cpp',opts);

movefile('*.cpp', output_dir);
movefile('*.h', output_dir);
% movefile('*.mex*', output_dir);

%%
% import casadi.*
% 
% output_dir = "gen_casadi_cpp_single";
% % ====== 定义变量略过 ======
% % 假设你已经定义了 x, u, y, z, s, etc.
% 
% xu = [x(:); u(:)];
% 
% % 所有函数统一参数
% args = {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd(:), ud(:), dt(:)};
% 
% % 用结构数组定义所有函数和表达式
% fun_defs = {
%     {'b_kkt',     {b}};
%     {'A_kkt',     {A}};
%     {'Asym_kkt',  {A_sym}};
%     {'f',         {f}};
%     {'df',        {df}};
%     {'hg',        {h, g}};
% };
% 
% for k = 1:length(fun_defs)
%     name = fun_defs{k}{1};
%     out_expr = fun_defs{k}{2};
% 
%     f = Function(name, args, out_expr);
% 
%     % cg = casadi.CodeGenerator([name '.cpp']);
%     cg = casadi.CodeGenerator([name '.cpp'], struct( ...
%     'with_header', true, ...
%     'precision', 'single' ...
% ));
%     % cg.setOption('with_header', true);
%     % cg.setOption('precision', 'single');
%     % cg.set_option('precision', 'single');  % ✅ 单精度
%     cg.add(f);
%     cg.generate();
% end
% 
% % ========= 特殊处理 Retraction =========
% 
% xu_new = [x_new(:); u_new(:)];
% f_ret = Function('Retraction_xyz', ...
%                  {xu, y(:), z(:), s(:), [ex(:); eu(:); ey(:); ez(:); es(:)]}, ...
%                  {xu_new, y_new(:), z_new(:), s_new(:)});
% 
% cg = casadi.CodeGenerator('Retraction_xyz.cpp');
% cg.set_option('with_header', true);
% cg.set_option('precision', 'single');
% cg.add(f_ret);
% cg.generate();
% 
% movefile('*.cpp', output_dir);
% movefile('*.h', output_dir);

%%
function w = Log_SO3(R)

t = (trace(R) - 1)/2;
t = if_else(t < -1, -1, t);
t = if_else(t >  1,  1, t);
t = acos( t );
w = t / 2 / sin( sqrt(abs(t^2) + 1e-100) ) * [R(3, 2) - R(2, 3);
                           R(1, 3) - R(3, 1);
                           R(2, 1) - R(1, 2)];
end

function R = Exp_SO3_Rodriguez(t)
tt = sqrt(sum(t.^2) + 1e-100);
R = eye(3) + sin(tt) / tt * skew_SO3(t) + ...
      (1 - cos(tt)) / (tt^2) * skew_SO3(t)^2;
end

% output_dir = "gen_casadi";
% 
% fun_eq = Function('eq_dynamics_mex',vars_fun, {eq});
% fun_eq.generate('eq_dynamics_mex.c',opts);
% 
% fun_eq_Jac = Function('Jac_dynamics_mex',vars_fun, {Jac_eq});
% fun_eq_Jac.generate('Jac_dynamics_mex.c',opts);
% 
% fun_eq_Hess = Function('Hess_dynamics_mex',vars_fun, {Hess_eq});
% fun_eq_Hess.generate('Hess_dynamics_mex.c',opts);
% 
% c_files = dir('*.c');
% for k = 1:length(c_files)
%     try
%         mex(c_files(k).name);
%     catch ME
%         warning('Failed to compile %s: %s', c_files(k).name, ME.message);
%     end
% end
% %%
% movefile('*.c', output_dir);
% % movefile('*.h', output_dir);            
% movefile('*.mex*', output_dir);

% import casadi.*
% 
% % Sparsity(A)
% 
% solver = Linsol('solver', 'qr', A.sparsity());
% 
% fun_b = Function('b_kkt_mex', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd, ud, dt}, {b});
% fun_A = Function('A_kkt_mex', {xu, y(:), z(:), s(:), mu, x0, Qstage, Qterminal, Rcontrol(:), xd, ud, dt}, {A});
% 
% bnum = full(fun_b( ...
%     randn(size(xu)), ...         % xu
%     randn(size(y(:))), ...       % y
%     randn(size(z(:))), ...       % z
%     randn(size(s(:))), ...       % s
%     1e-2, ...                    % mu
%     randn(size(x0)), ...         % x0
%     randn(size(Qstage)), ...     % Qstage
%     randn(size(Qterminal)), ...  % Qterminal
%     randn(size(Rcontrol(:))), ...% Rcontrol
%     randn(size(xd)), ...         % xd
%     randn(size(ud)), ...         % ud
%     randn(size(dt)) ...          % dt
% ));
% 
% Anum = fun_A( ...
%     randn(size(xu)), ...         % xu
%     randn(size(y(:))), ...       % y
%     randn(size(z(:))), ...       % z
%     randn(size(s(:))), ...       % s
%     1e-2, ...                    % mu
%     randn(size(x0)), ...         % x0
%     randn(size(Qstage)), ...     % Qstage
%     randn(size(Qterminal)), ...  % Qterminal
%     randn(size(Rcontrol(:))), ...% Rcontrol
%     randn(size(xd)), ...         % xd
%     randn(size(ud)), ...         % ud
%     randn(size(dt)) ...          % dt
% );
% 
% tic
% solver.solve(A, b)
% toc
% solver.set_input(Sparsity(A), 'sp_A')
