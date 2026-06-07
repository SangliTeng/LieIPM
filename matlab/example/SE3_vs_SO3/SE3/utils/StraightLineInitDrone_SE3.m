function X = StraightLineInitDrone_SE3(X0, Xg, Ns)
% Initialize by straight-line interpolation on SE(3)
% No BatchRetractSO3xR3 is used.

X = cell(Ns + 1, 1);
X{1} = X0;

for i = 1:X0.num_srb
    R0{i} = X0.mrb(i).R;
    p0{i} = X0.mrb(i).p;
    Rg{i} = Xg.mrb(i).R;
    pg{i} = Xg.mrb(i).p;

    T0 = [R0{i}, p0{i};
          0 0 0 1];
    Tg = [Rg{i}, pg{i};
          0 0 0 1];

    Xi{i} = Log_SE3(T0 \ Tg);   % body-frame twist, 6x1
end

for k = 2:Ns+1
    alpha = (k - 1) / Ns;
    X{k} = X0;

    for i = 1:X0.num_srb
        T0 = [R0{i}, p0{i};
              0 0 0 1];

        Tinterp = T0 * Exp_SE3(alpha * Xi{i});

        X{k}.mrb(i).R = Tinterp(1:3, 1:3);
        X{k}.mrb(i).p = Tinterp(1:3, 4);
    end
end

end


function xi = Log_SE3(T)
R = T(1:3, 1:3);
p = T(1:3, 4);

phi = Log_SO3(R);
theta = norm(phi);

if theta < 1e-12
    Vinv = eye(3) - 0.5 * skew_SO3(phi) + (1/12) * skew_SO3(phi)^2;
else
    A = skew_SO3(phi);
    half_theta = 0.5 * theta;
    cot_half = cos(half_theta) / sin(half_theta);
    Vinv = eye(3) - 0.5 * A ...
         + (1/theta^2) * (1 - half_theta * cot_half) * (A^2);
end

rho = Vinv * p;
xi = [phi; rho];
end


function T = Exp_SE3(xi)
phi = xi(1:3);
rho = xi(4:6);

R = Exp_SO3(phi);
theta = norm(phi);

if theta < 1e-12
    V = eye(3) + 0.5 * skew_SO3(phi) + (1/6) * skew_SO3(phi)^2;
else
    A = skew_SO3(phi);
    V = eye(3) ...
      + (1 - cos(theta)) / theta^2 * A ...
      + (theta - sin(theta)) / theta^3 * (A^2);
end

p = V * rho;

T = [R, p;
     0 0 0 1];
end


function R = Exp_SO3(phi)
theta = norm(phi);
A = skew_SO3(phi);

if theta < 1e-12
    R = eye(3) + A + 0.5 * A^2;
else
    R = eye(3) ...
      + sin(theta) / theta * A ...
      + (1 - cos(theta)) / theta^2 * A^2;
end
end


function S = skew_SO3(w)
S = [    0, -w(3),  w(2);
      w(3),     0, -w(1);
     -w(2),  w(1),     0];
end