function [eq, gvec] = gen_SO3xR3_EqualityConstraints(vars, cons_list, inertial_list, g, input_map)
import casadi.*

num_srb = length(inertial_list);
lam = vars.lam;
elam = vars.elam;

xmat = vars.xmat;
emat = vars.emat;

u    = vars.u;
eu   = vars.eu;

dt = vars.dt;

%% generate the perturbed SO(3) x R3 elements for input map design
xmat_eps = [];
xmat_kp1_eps = [];

for k = 1:num_srb
    xk_eps = [];
    xkp1   = [];
    for n = 1:2
        R = xmat((1:9)    + 24 * (n - 1), k); R = reshape(R, [3, 3]);
        p = xmat((10:12)  + 24 * (n - 1), k);
        F = xmat((13:21)  + 24 * (n - 1), k); F = reshape(F, [3, 3]);
        v = xmat((22:24)  + 24 * (n - 1), k);

        eR = emat((1:3)   + 12 * (n - 1), k);
        ep = emat((4:6)   + 12 * (n - 1), k);
        eF = emat((7:9)   + 12 * (n - 1), k);
        ev = emat((10:12) + 12 * (n - 1), k);

        Reps   = R * Exp_SO3(eR);
        peps   = p + ep;
        Feps   = F * Exp_SO3(eF);
        veps   = v + ev;

        Rkp1 = Reps * Feps;
        pkp1 = peps + veps * dt;

        xk_eps = [xk_eps; Reps(:); peps(:); Feps(:); veps(:)];
        xkp1   = [  xkp1; Rkp1(:); pkp1(:);    F(:);    v(:)];
    end
    
    xmat_eps     = [xmat_eps,   xk_eps];
    xmat_kp1_eps = [xmat_kp1_eps, xkp1];
end

%% generate the second-order retraction for rigid body dynamics
z_eps = [];
for k = 1:num_srb
    R1 = xmat(1:9, k); R1 = reshape(R1, [3, 3]);
    p1 = xmat(10:12, k);
    F1 = xmat(13:21, k); F1 = reshape(F1, [3, 3]);
    v1 = xmat(22:24, k);

    R2 = xmat((1:9) + 24, k); R2 = reshape(R2, [3, 3]);
    p2 = xmat((10:12) + 24, k);
    F2 = xmat((13:21) + 24, k); F2 = reshape(F2, [3, 3]);
    v2 = xmat((22:24) + 24, k);

    eR1 = emat(1:3, k);
    ep1 = emat(4:6, k);
    eF1 = emat(7:9, k);
    ev1 = emat(10:12, k);

    eR2 = emat((1:3)+12, k);
    ep2 = emat((4:6)+12, k);
    eF2 = emat((7:9)+12, k);
    ev2 = emat((10:12)+12, k);

    Jd = inertial_list{k}.Jd;
    m  = inertial_list{k}.m;
    % fR = R2' * R1 * F1;
    Y = R2' * R1 * F1;

    zR = 1 / 2 * skew_SO3(-Y' * eR2) * (F1' * eR1);
    zR = 1 / 2 * skew_SO3(-Y' * eR2) * (eye(3) * eF1) + zR;
    zR = 1 / 2 * skew_SO3(F1' * eR1) * (eye(3) * eF1) + zR;
    zR = zR - Y' * eR2 + F1' * eR1 + eye(3) * eF1;
    
    % consider add the offset
    zR = zR + Log_SO3(Y);
    

    zp = (p2 + ep2) - ((p1+ep1) + (v1+ev1) * dt);

    M2 = F2 * Exp_SO3(eF2) * Jd - Jd' * Exp_SO3(-eF2) * F2';
    M1 = Jd * F1 * Exp_SO3(eF1) - Exp_SO3(-eF1) * F1' * Jd';

    zF = vee_SO3(M2 - M1) / dt;

    zv = m * ((v2 + ev2) - (v1 + ev1) - g * dt);

    % z = [z; zR; zp; zF; zv];
    z_eps = [z_eps; zR; zp; zF; zv];
end

%% Generate the second-order correction for the constraints
gvec_eps = [];
for k = 1:length(cons_list)
    id1 = cons_list{k}.id_parent;
    id2 = cons_list{k}.id_child;

    if id1 > 0
        eR = emat((1:3)+12, id1);
        ep = emat((4:6)+12, id1);

        R1 = xmat((1:9) + 24, id1); 
        R1 = reshape(R1, [3, 3]);

        R1 = R1 * Exp_SO3(eR);
        p1 = xmat((10:12) + 24, id1) + ep;
    else
        R1 = [];
        p1 = [];
    end

    if id2 > 0
        eR = emat((1:3)+12, id2);
        ep = emat((4:6)+12, id2);

        R2 = xmat((1:9) + 24, id2); 
        R2 = reshape(R2, [3, 3]);

        R2 = R2 * Exp_SO3(eR);
        p2 = xmat((10:12) + 24, id2) + ep;
    else
        R2 = [];
        p2 = [];
    end
    % [G1, G2] = cons_list{k}.getJacobians(R1, p1, R2, p2);
    gvec_eps = [gvec_eps;
            cons_list{k}.getValues(R1, p1, R2, p2)];
end

opts = struct(); % 'allow_free',true);
% opts.allow_free=true;
emat_2 = emat(13:end, :);
G = jacobian(gvec_eps, emat_2(:)); % this will be the input map for the holonomic constraints
G_fun = Function('G', {emat_2(:)}, {G}, opts);
G = G_fun(emat_2(:) * 0);
G = [zeros(size(G, 1), 6), G(:, 1:end-6)];
Glam = G' * lam;
fun_Glam = Function('Glam', {xmat, lam}, {Glam}); % need to obtain the second-order retraction of the 
Glam_eps = fun_Glam(xmat_eps, lam + elam);
%%
fun_input = Function('input', {xmat, vars.u}, {input_map});
input_map_eps = fun_input(xmat_eps, u + eu);
%%
eq = [z_eps - Glam_eps * dt - input_map_eps * dt;
      gvec_eps];
%%
fun_gvec = Function('gvec', {emat_2(:)}, {gvec_eps});
gvec = fun_gvec(emat_2(:) * 0);
fun_gvec = Function('gvec', {xmat}, {gvec});
gvec = fun_gvec(xmat_kp1_eps);
end

function w = Log_SO3(R)

t = (trace(R) - 1)/2;
t = if_else(t < -1, -1, t);
t = if_else(t >  1,  1, t);
t = acos( t );
w = t / 2 / sin( sqrt(abs(t^2) + 1e-100) ) * [R(3, 2) - R(2, 3);
                           R(1, 3) - R(3, 1);
                           R(2, 1) - R(1, 2)];
end