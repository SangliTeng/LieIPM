function [stage_cost, vars] = gen_SE3_trace_cost()
import casadi.*

% -------------------------------------------------------------------------
% States / desired states
% -------------------------------------------------------------------------
R  = SX.sym('R',  3, 3);
Rd = SX.sym('Rd', 3, 3);

F  = SX.sym('F',  3, 3);
Fd = SX.sym('Fd', 3, 3);

p  = SX.sym('p',  3, 1);
pd = SX.sym('pd', 3, 1);

v  = SX.sym('v',  3, 1);
vd = SX.sym('vd', 3, 1);

% -------------------------------------------------------------------------
% Raw symbolic weights
% -------------------------------------------------------------------------
QR_sym = SX.sym('QR', 3, 3);
Qp_sym = SX.sym('Qp', 3, 3);
QF_sym = SX.sym('QF', 3, 3);
Qv_sym = SX.sym('Qv', 3, 3);

% Symmetrized versions used ONLY in cost construction
QR = 0.5 * (QR_sym + QR_sym');
Qp = 0.5 * (Qp_sym + Qp_sym');
QF = 0.5 * (QF_sym + QF_sym');
Qv = 0.5 * (Qv_sym + Qv_sym');

% -------------------------------------------------------------------------
% Relative SE(3) errors
% -------------------------------------------------------------------------
R_rel = R' * Rd;
% p_rel = R' * (pd - p);
p_rel = (pd - p);

F_rel = F' * Fd;
% v_rel = F' * (vd - v);
v_rel = (vd - v);

RR  = SX.sym('RR', [3,3]);
Q   = SX.sym('Q',  [3,3]);
Q_sym = 0.5 * (Q + Q');
rot_loss = 0.5 * trace((RR - eye(3)) * Q_sym * (RR - eye(3))');

stage_cost = ...
    substitute(rot_loss, [RR(:); Q(:)], [R_rel(:); QR(:)]) + ...
    0.5 * p_rel' * Qp * p_rel + ...
    substitute(rot_loss, [RR(:); Q(:)], [F_rel(:); QF(:)]) + ...
    0.5 * v_rel' * Qv * v_rel;

% perturbation
e = SX.sym('e', [12,1]);
eR = e(1:3);
ep = e(4:6);
eF = e(7:9);
ev = e(10:12);

% [R_eps, p_eps] = Retract_SE3_Left_2nd(R, p, eR, ep);
% [F_eps, v_eps] = Retract_SE3_Left_2nd(F, v, eF, ev);

eRhat = skew_SO3(eR);
eFhat = skew_SO3(eF);

R_eps = R * (eye(3) + eRhat + 0.5 * (eRhat * eRhat));
p_eps = p + R * (ep + 0.5 * eRhat * ep);

F_eps = F * (eye(3) + eFhat + 0.5 * (eFhat * eFhat));
v_eps = v + F * (ev + 0.5 * eFhat * ev);

stage_cost = substitute(stage_cost, R, R_eps);
stage_cost = substitute(stage_cost, p, p_eps);
stage_cost = substitute(stage_cost, F, F_eps);
stage_cost = substitute(stage_cost, v, v_eps);

vars.x  = [R(:); p(:); F(:); v(:)];
vars.xd = [Rd(:); pd(:); Fd(:); vd(:)];

% IMPORTANT: expose raw symbolic variables, not symmetrized expressions
vars.QR = QR_sym;
vars.Qp = Qp_sym;
vars.QF = QF_sym;
vars.Qv = Qv_sym;

vars.e = e;
end

function [R_new, p_new] = Retract_SE3_Left_2nd(R, p, phi, rho)
    dX = [skew_SO3(phi), rho;
          zeros(1,3), 0];

    Xnew = [R, p;
            0, 0, 0, 1] * (eye(4) + dX + 0.5 * (dX*dX) + (1/6) * (dX*dX*dX));

    R_new = Xnew(1:3,1:3);
    p_new = Xnew(1:3,4);
end

function S = skew_SO3(w)
    S = [    0, -w(3),  w(2);
          w(3),     0, -w(1);
         -w(2),  w(1),     0];
end