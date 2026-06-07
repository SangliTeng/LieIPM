function [eq, gvec] = gen_SE3_EqualityConstraints(vars, cons_list, inertial_list, g, input_map)
import casadi.*

num_srb = length(inertial_list);
lam  = vars.lam;
elam = vars.elam;

xmat = vars.xmat;
emat = vars.emat;

u  = vars.u;
eu = vars.eu;

dt = vars.dt;

%% ------------------------------------------------------------------------
% Full SE(3) retraction on both blocks:
%   X_k = (R_k, p_k) in SE(3)
%   G_k = (F_k, v_k) in SE(3)
%
% State packing is kept unchanged for compatibility:
%   [R(:); p; F(:); v]
% but the perturbation is interpreted as
%   [xi_X; xi_G] = [eR; ep; eF; ev] in se(3) x se(3)
%% ------------------------------------------------------------------------

%% generate the perturbed SE(3) x SE(3) elements for input map design
xmat_eps     = [];
xmat_kp1_eps = [];

for k = 1:num_srb
    xk_eps = [];
    xkp1   = [];

    for n = 1:2
        R = xmat((1:9)    + 24 * (n - 1), k);  R = reshape(R, [3, 3]);
        p = xmat((10:12)  + 24 * (n - 1), k);
        F = xmat((13:21)  + 24 * (n - 1), k);  F = reshape(F, [3, 3]);
        v = xmat((22:24)  + 24 * (n - 1), k);

        eR = emat((1:3)   + 12 * (n - 1), k);
        ep = emat((4:6)   + 12 * (n - 1), k);
        eF = emat((7:9)   + 12 * (n - 1), k);
        ev = emat((10:12) + 12 * (n - 1), k);

        [Reps, peps] = Retract_SE3_Left(R, p, eR, ep);
        [Feps, veps] = Retract_SE3_Left(F, v, eF, ev);

        xk_eps = [xk_eps; Reps(:); peps(:); Feps(:); veps(:)];

        [Rkp1, pkp1] = Mul_SE3(Reps, peps, Feps, veps);
        xkp1 = [xkp1; Rkp1(:); pkp1(:); Feps(:); veps(:)];
    end

    xmat_eps     = [xmat_eps, xk_eps];
    xmat_kp1_eps = [xmat_kp1_eps, xkp1];
end

%% generate the retracted local second-order equality model
z_eps = [];
for k = 1:num_srb
    % ----- stage 1 -----
    R1 = xmat(1:9, k);          R1 = reshape(R1, [3, 3]);
    p1 = xmat(10:12, k);
    F1 = xmat(13:21, k);        F1 = reshape(F1, [3, 3]);
    v1 = xmat(22:24, k);

    % ----- stage 2 -----
    R2 = xmat((1:9) + 24, k);   R2 = reshape(R2, [3, 3]);
    p2 = xmat((10:12) + 24, k);
    F2 = xmat((13:21) + 24, k); F2 = reshape(F2, [3, 3]);
    v2 = xmat((22:24) + 24, k);

    % ----- perturbations -----
    eR1 = emat(1:3, k);
    ep1 = emat(4:6, k);
    eF1 = emat(7:9, k);
    ev1 = emat(10:12, k);

    eR2 = emat((1:3) + 12, k);
    ep2 = emat((4:6) + 12, k);
    eF2 = emat((7:9) + 12, k);
    ev2 = emat((10:12) + 12, k);

    Jd = inertial_list{k}.Jd;
    m  = inertial_list{k}.m;

    %% (1) kinematics residual split into rotation / translation
    % operating point for the rotation-chain constraint
    Y = R2' * R1 * F1;

    % constant residuals at the operating point
    zR0 = Log_SO3(Y);
    zp0 = p2 - p1 - R1 * v1 * dt;

    % rotation: low-order to high-order
    % zR = zR0 ...
    %      - Y' * eR2 ...
    %      + F1' * eR1 ...
    %      + eF1 ...
    %      + 0.5 * vee_SO3(skew_SO3(-Y' * eR2) * skew_SO3(F1' * eR1) - skew_SO3(F1' * eR1) * skew_SO3(-Y' * eR2)) ...
    %      + 0.5 * vee_SO3(skew_SO3(-Y' * eR2) * skew_SO3(eF1)       - skew_SO3(eF1)       * skew_SO3(-Y' * eR2)) ...
    %      + 0.5 * vee_SO3(skew_SO3(F1' * eR1) * skew_SO3(eF1)       - skew_SO3(eF1)       * skew_SO3(F1' * eR1));

    zR = 1 / 2 * skew_SO3(-Y' * eR2) * (F1' * eR1);
    zR = 1 / 2 * skew_SO3(-Y' * eR2) * (eye(3) * eF1) + zR;
    zR = 1 / 2 * skew_SO3(F1' * eR1) * (eye(3) * eF1) + zR;
    zR = zR - Y' * eR2 + F1' * eR1 + eye(3) * eF1;
    
    % consider add the offset
    zR = zR + Log_SO3(Y);


    % translation: low-order to high-order
    zp = zp0 ...
         + ep2 ...
         - F1' * ep1 ...
         - dt * ev1 ...
         - dt * F1' * skew_SO3(eR1) * v1 ...
         + 0.5 * skew_SO3(eR2) * ep2 ...
         - 0.5 * F1' * skew_SO3(eR1) * ep1 ...
         - dt * F1' * skew_SO3(eR1) * F1 * ev1 ...
         - 0.5 * dt * skew_SO3(eF1) * ev1 ...
         - 0.5 * dt * F1' * (skew_SO3(eR1) * skew_SO3(eR1)) * v1;

    %% (2) rotational dynamics residual
    zF0 = vee_SO3(F2 * Jd - Jd * F2' - Jd * F1 + F1' * Jd) / dt;

    zF = zF0 ...
         + vee_SO3(F2 * skew_SO3(eF2) * Jd + Jd * skew_SO3(eF2) * F2') / dt ...
         - vee_SO3(F1 * skew_SO3(eF1) * Jd + Jd * skew_SO3(eF1) * F1') / dt ...
         + 0.5 * vee_SO3(F2 * (skew_SO3(eF2) * skew_SO3(eF2)) * Jd ...
                         - Jd * (skew_SO3(eF2) * skew_SO3(eF2)) * F2') / dt ...
         - 0.5 * vee_SO3(F1 * (skew_SO3(eF1) * skew_SO3(eF1)) * Jd ...
                         - Jd * (skew_SO3(eF1) * skew_SO3(eF1)) * F1') / dt;

    %% (3) translational dynamics residual
    zv0 = m * (v2 - F1' * v1 - R2' * g * dt);

    zv = zv0 ...
         + m * F2 * ev2 ...
         - m * ev1 ...
         + m * skew_SO3(eF1) * F1' * v1 ...
         + m * dt * skew_SO3(eR2) * R2' * g ...
         + 0.5 * m * F2 * skew_SO3(eF2) * ev2 ...
         + 0.5 * m * skew_SO3(eF1) * ev1 ...
         - 0.5 * m * (skew_SO3(eF1) * skew_SO3(eF1)) * F1' * v1 ...
         - 0.5 * m * dt * (skew_SO3(eR2) * skew_SO3(eR2)) * R2' * g;

    z_eps = [z_eps; zR; zp; zF; zv];
end

%% Generate the perturbed holonomic constraints at the next pose
% only the stage-2 pose enters the holonomic constraints.
gvec_eps = [];
for k = 1:length(cons_list)
    id1 = cons_list{k}.id_parent;
    id2 = cons_list{k}.id_child;

    if id1 > 0
        eR = emat((1:3) + 12, id1);
        ep = emat((4:6) + 12, id1);

        R1 = xmat((1:9) + 24, id1);
        R1 = reshape(R1, [3, 3]);
        p1 = xmat((10:12) + 24, id1);

        [R1, p1] = Retract_SE3_Left(R1, p1, eR, ep);
    else
        R1 = [];
        p1 = [];
    end

    if id2 > 0
        eR = emat((1:3) + 12, id2);
        ep = emat((4:6) + 12, id2);

        R2 = xmat((1:9) + 24, id2);
        R2 = reshape(R2, [3, 3]);
        p2 = xmat((10:12) + 24, id2);

        [R2, p2] = Retract_SE3_Left(R2, p2, eR, ep);
    else
        R2 = [];
        p2 = [];
    end

    gvec_eps = [gvec_eps;
                cons_list{k}.getValues(R1, p1, R2, p2)];
end

opts = struct();
emat_2 = emat(13:end, :);
G = jacobian(gvec_eps, emat_2(:));
G_fun = Function('G', {emat_2(:)}, {G}, opts);
G = G_fun(emat_2(:) * 0);

G = [zeros(size(G, 1), 6), G(:, 1:end-6)];
Glam = G' * lam;
fun_Glam = Function('Glam', {xmat, lam}, {Glam});
Glam_eps = fun_Glam(xmat_eps, lam + elam);

fun_input = Function('input', {xmat, vars.u}, {input_map});
input_map_eps = fun_input(xmat_eps, u + eu);

%% equality constraints
eq = [z_eps - Glam_eps * dt - input_map_eps * dt;
      gvec_eps];

%% next-step holonomic constraints for terminal use
gvec_zero_fun = Function('gvec_zero_fun', {emat_2(:)}, {gvec_eps});
gvec = gvec_zero_fun(emat_2(:) * 0);

gvec_x_fun = Function('gvec_x_fun', {xmat}, {gvec});
gvec = gvec_x_fun(xmat_kp1_eps);
end

%% =========================================================================
function [Rnew, pnew] = Retract_SE3_Left(R, p, phi, rho)
J = LeftJacobian_SO3(phi);
Rnew = R * Exp_SO3_Rodriguez(phi);
pnew = p + R * J * rho;
end

function [R3, p3] = Mul_SE3(R1, p1, R2, p2)
R3 = R1 * R2;
p3 = p1 + R1 * p2;
end
% 
% function [Rinv, pinv] = Inv_SE3(R, p)
% Rinv = R';
% pinv = -R' * p;
% end

function J = LeftJacobian_SO3(phi)
tt = sqrt(sum(phi.^2) + 1e-6);
phihat = skew_SO3(phi);
A = (1 - cos(tt)) / (tt^2);
B = (tt - sin(tt)) / (tt^3);
J = eye(3) + A * phihat + B * (phihat * phihat);
end

% function R = Exp_SO3_Rodriguez(t)
% tt = sqrt(sum(t.^2) + 1e-6);
% K = skew_SO3(t);
% R = eye(3) + sin(tt) / tt * K + (1 - cos(tt)) / (tt^2) * (K * K);
% end
function R = Exp_SO3_Rodriguez(t)
tt = sqrt(sum(t.^2) + 1e-6);
R = eye(3) + sin(tt) / tt * skew_SO3(t) + ...
      (1 - cos(tt)) / (tt^2) * skew_SO3(t)^2;
end

% function R = Exp_SO3_Rodriguez(t)
% tt2 = sum(t.^2);
% K = skew_SO3(t);
% 
% A = if_else(tt2 < 1e-8, 1 - tt2/6 + tt2^2/120, sin(sqrt(tt2))/sqrt(tt2));
% B = if_else(tt2 < 1e-8, 1/2 - tt2/24 + tt2^2/720, (1 - cos(sqrt(tt2)))/tt2);
% 
% R = eye(3) + A * K + B * (K * K);
% end

% function w = Log_SO3(R)
% t = (trace(R) - 1) / 2;
% t = if_else(t < -1, -1, t);
% t = if_else(t >  1,  1, t);
% t = acos(t);
% 
% w = t / (2 * sin(sqrt(abs(t^2) + 1e-6))) * ...
%     [R(3, 2) - R(2, 3);
%      R(1, 3) - R(3, 1);
%      R(2, 1) - R(1, 2)];
% end

% function S = skew_SO3(w)
% S = [    0, -w(3),  w(2);
%       w(3),     0, -w(1);
%      -w(2),  w(1),     0];
% end

% function v = vee_SO3(S)
% v = [S(3, 2); S(1, 3); S(2, 1)];
% end
