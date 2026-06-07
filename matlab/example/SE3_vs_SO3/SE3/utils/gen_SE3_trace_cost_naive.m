function [stage_cost, vars] = gen_SE3_trace_cost_naive()
import casadi.*

RR  = SX.sym('RR',  [3, 3]);
RRd = SX.sym('RRd', [3, 3]);
Q   = SX.sym('Q',   [3, 3]);

Q_sym = (Q' + Q) / 2;
f = 1 / 2 * trace((RR * RRd' - eye(3)) * Q_sym * (RR * RRd' - eye(3))');

% 3x3 SO(3)-related matrices
R  = SX.sym('R',  3, 3);
Rd = SX.sym('Rd', 3, 3);
F  = SX.sym('F',  3, 3);
Fd = SX.sym('Fd', 3, 3);

% Position and velocity
p  = SX.sym('p',  3, 1);
pd = SX.sym('pd', 3, 1);
v  = SX.sym('v',  3, 1);
vd = SX.sym('vd', 3, 1);

% Weights
QR = SX.sym('QR', 3, 3);
QF = SX.sym('QF', 3, 3);
Qp = SX.sym('Qp', 3, 3);
Qv = SX.sym('Qv', 3, 3);

stage_cost = ...
    substitute(f, [RR(:); RRd(:); Q(:)], [R(:); Rd(:); QR(:)]) + ...
    substitute(f, [RR(:); RRd(:); Q(:)], [F(:); Fd(:); QF(:)]) + ...
    1 / 2 * (v - vd)' * Qv * (v - vd) + ...
    1 / 2 * (p - pd)' * Qp * (p - pd);

e = SX.sym('e', [12, 1]);

eR = e(1:3);
ep = e(4:6);
eF = e(7:9);
ev = e(10:12);

eRhat = skew_SO3(eR);
eFhat = skew_SO3(eF);

% SE(3) second-order retraction
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

vars.QR = QR;
vars.Qp = Qp;
vars.QF = QF;
vars.Qv = Qv;

vars.e = e;
end

function S = skew_SO3(w)
S = [    0, -w(3),  w(2);
      w(3),     0, -w(1);
     -w(2),  w(1),     0];
end