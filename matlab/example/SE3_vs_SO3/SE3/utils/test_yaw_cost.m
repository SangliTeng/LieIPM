clear; clc;

N = 10;
eps_smooth = 1e-12;
e1 = [1;0;0]; e2 = [0;1;0]; e3 = [0;0;1];
E  = [e1 e2];

fprintf('%5s %12s %12s %12s %12s\n','k','yaw(Q)','yaw(S)','diffQS','yaw_ZYX(R)');
for k = 1:N
    % 随机 R
    axis = randn(3,1); axis = axis/norm(axis);
    ang  = 2*pi*rand;
    K = [0,-axis(3),axis(2); axis(3),0,-axis(1); -axis(2),axis(1),0];
    R = expm(K*ang);

    % 分解：R -> (Q, n), 以及构造基
    n = R*e3;
    Pn = eye(3) - n*n.';
    w = Pn*e1; u = Pn*e2;
    a = sqrt(w.'*w + eps_smooth^2);
    b = sqrt(u.'*u + eps_smooth^2);
    v = a*w + b*u;
    t1 = v / max(norm(v), eps_smooth);
    t2 = cross(n, t1);
    T  = [t1 t2];

    Q = T.' * R * E;                   % 绕 n 的 SO(2)（残余 yaw）
    yaw_Q = atan2(Q(2,1), Q(1,1));

    % 正确的残余矩阵（应为 blkdiag(Q,1)）
    A = [T n] * [E e3].';              % 这是把 e3 送到 n 的等距对齐
    S = A.' * R;                        % 去掉 roll/pitch 后的残余
    yaw_S = atan2(S(2,1), S(1,1));      % 绕世界 e3 的角度，但此时 S*e3=e3

    % 对比：yaw(Q) 应等于 yaw(S)
    diffQS = wrapToPi(yaw_Q - yaw_S);

    % 另打印标准 ZYX 世界系 yaw（一般 != yaw_Q）
    yaw_ZYX = atan2(R(2,1), R(1,1));

    fprintf('%5d %12.6f %12.6f %12.6f %12.6f\n', k, yaw_Q, yaw_S, diffQS, yaw_ZYX);

    % 额外一致性检查（可选）
    R_rec = [T n] * blkdiag(Q,1) * [E e3].';
    assert(norm(R - R_rec, 'fro') < 1e-10);
    assert(norm(S - blkdiag(Q,1), 'fro') < 1e-10);
end
