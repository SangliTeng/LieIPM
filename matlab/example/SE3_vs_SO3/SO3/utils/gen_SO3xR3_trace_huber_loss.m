function [stage_cost, vars] = gen_SO3xR3_trace_huber_loss()
import casadi.*
RR  = SX.sym('RR',  [3, 3]);
RRd = SX.sym('RRd', [3, 3]);
Q   = SX.sym('Q',   [3, 3]);

Q_sym = (Q' + Q) / 2;

% R_eps = RR; % * Exp_SO3(e);
f   = 1 / 2 * trace( (RR * RRd' - eye(3)) * Q_sym * (RR * RRd' - eye(3))' );

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

% Control weight and control input
W = SX.sym('W', 6, 6);
u = SX.sym('u', 6, 1);

beta = 1;
stage_cost = substitute(f, [RR(:); RRd(:); Q(:)], [R(:); Rd(:); QR(:)]) + ... %  + subs(f, [e; Q(:)], [eF; QF(:)]) + ...
    substitute(f, [RR(:); RRd(:); Q(:)], [F(:); Fd(:); QF(:)]) + ...
    huber_loss(sqrt((v - vd)' * Qv * (v - vd)), beta) + ...
    huber_loss(sqrt((p - pd)' * Qp * (p - pd)), beta); % + ...
    % smooth_sqrt((v - vd)' * Qv * (v - vd)) + smooth_sqrt((p - pd)' * Qp * (p - pd)); % + ...

e   = SX.sym('e',   [12, 1]);

stage_cost = substitute(stage_cost,  R, R * Exp_SO3(e(1:3)));
stage_cost = substitute(stage_cost,  p, p + e(4:6));
stage_cost = substitute(stage_cost,  F, F * Exp_SO3(e(7:9)));
stage_cost = substitute(stage_cost,  v, v + (e(10:12)));

% vars.x.R = R;
% vars.x.p = p;
% vars.x.F = F;
% vars.x.v = v;
% 
% vars.xd.Rd = Rd;
% vars.xd.pd = pd;
% vars.xd.Fd = Fd;
% vars.xd.vd = vd;

vars.x  = [R(:); p(:); F(:); v(:)];
vars.xd = [Rd(:); pd(:); Fd(:); vd(:)];

vars.QR = QR;
vars.Qp = Qp;
vars.QF = QF;
vars.Qv = Qv;

vars.e = e;
end
%%
function t = huber_loss(t, beta)
    % t = sqrt(abs(t)^2 + 1e-2);
    t = if_else(abs(t) < beta, t^2 / 2 / beta^2, abs(t) / beta - 1 / 2);
end