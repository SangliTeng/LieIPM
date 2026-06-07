function [stage_cost, vars] = gen_SO3xR3_flat_cost()
import casadi.*
RR  = SX.sym('RR',  [3, 3]);
RRd = SX.sym('RRd', [3, 3]); % Will only import the yaw angles... 
Q   = SX.sym('Q',   [3, 3]);

Q_sym = (Q' + Q) / 2;

% R_eps = RR; % * Exp_SO3(e);
% f   = 1 / 2 * trace( (RR * RRd' - eye(3)) * Q_sym * (RR * RRd' - eye(3))' );
% sin(x - y) = sx * cy - cx * sy
n0 = sqrt(RR(2, 1)^2 + RR(1, 1)^2 + 1e-10);
fR = (RR(2, 1) / n0 * RRd(1, 1) - RR(1, 1) / n0 * RRd(2, 1))^2 * Q_sym(1, 1) / 2;
fR = fR + 1e-3 * trace( (eye(3) - RR * RRd') * (eye(3) - RR * RRd')' );

% 3x3 SO(3)-related matrices
R  = SX.sym('R',  3, 3);
Rd = SX.sym('Rd', 3, 3);
F  = SX.sym('F',  3, 3);
Fd = SX.sym('Fd', 3, 3);

R2  = R * F;
c1  =  R(1, 1); s1  =  R(1, 2); n1 = sqrt(c1^2 + s1^2 + 1e-10);
c2  = R2(1, 1); s2  = R2(1, 2); n2 = sqrt(c2^2 + s2^2 + 1e-10);
c1 = c1 / n1; s1 = s1 / n1;
c2 = c2 / n2; s2 = s2 / n2;

ds  = s2 * c1 - c2 * s1; % sin(x - y) = sx * cy - cx * sy
dc  = c2 * c1 + s1 * s2; % cos(x - y) = cx * cy + sx * sy
fdR = (ds * Fd(1, 1) - dc * Fd(2, 1))^2 * Q_sym(1, 1) / 2;
fdR = fdR + 1e-4 * trace( (eye(3) - F) * (eye(3) - F)' );

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


stage_cost = substitute(fR, [RR(:); RRd(:); Q(:)], [R(:); Rd(:); QR(:)]) + ... %  + subs(f, [e; Q(:)], [eF; QF(:)]) + ...
    substitute(fdR, [Q(:)], [QF(:)]) + ...
    1 / 2 * (v - vd)' * Qv * (v - vd) + 1 / 2 * (p - pd)' * Qp * (p - pd); % + ...

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