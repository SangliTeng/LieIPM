function vars = gen_SO3xR3_Variables(cons_list, inertial_list, dim_u)
import casadi.*

num_srb = length(inertial_list);

dt   = SX.sym('dt');
xmat = SX.sym('x', [(12 + 12) * 2, num_srb]); % [R(:); p(:); F(:); v(:)]
emat = SX.sym('e', [12 * 2, num_srb]); 

emat_temporal = [emat(1:12, :), emat(13:end, :)]; % [x1; x2]

dof = 0; % dof of constraints
for k = 1:length(cons_list)
    dof = cons_list{k}.dof + dof;
end

u    = SX.sym('u',  [dim_u, 1]);
eu   = SX.sym('eu', [dim_u, 1]);
lam  = SX.sym('lam',  [dof, 1]); 
elam = SX.sym('elam', [dof, 1]);
% nu   = SX.sym('nu',  [length(emat(:)) / 2 + length(lam), 1]); % multipliers
nu   = SX.sym('nu',  [length(emat(:)) / 2 + dof, 1]); % multipliers

vars.xmat = xmat;
vars.emat = emat;
vars.emat_temporal = emat_temporal;

vars.lam  = lam;
vars.elam = elam;

vars.u  = u;
vars.eu = eu;

vars.dt = dt;

vars.nu = nu;
end