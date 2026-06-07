% Extract the SO(2) fiber (roll about body-x) from R ∈ SO(3)
% R should be ZYX: R = Rz*Ry*Rx; here we recover (cos phi, sin phi)
% such that R = [ s, c*u + s*v, -s*u + c*v ], with s = R*e1.
% --- quick self-check ---
z = randn; y = randn; x = randn;
Rz = [cos(z) -sin(z) 0; sin(z) cos(z) 0; 0 0 1];
Ry = [cos(y) 0 sin(y); 0 1 0; -sin(y) 0 cos(y)];
Rx = [1 0 0; 0 cos(x) -sin(x); 0 sin(x) cos(x)];
R  = Rx*Ry*Rz;
[c,s,sv,u,v] = so2_from_R_smooth(R);
Rhat = [sv, c*u + s*v, -s*u + c*v];
fprintf('err=%.e,  cos≈%.6f  sin≈%.6f\n', norm(R-Rhat,'fro'), c, s);

function [c, s, svec, u, v] = so2_from_R(R)
    svec = R(:,1);                         % S^2 coordinate
    a = [0;0;1]; if abs(svec(3))>=0.9, a=[0;1;0]; end
    u = cross(a, svec); u = u/norm(u);     % unit, ⟂ s
    v = cross(svec, u);                    % completes right-handed frame
    c = dot(R(:,2), u);                    % cos(phi)
    s = dot(R(:,2), v);                    % sin(phi)
end

function [cphi, sphi, svec, u, v] = so2_from_R_smooth(R)
% Extract smooth SO(2) (roll about s) from R∈SO(3) without branching.
% R = [s, c*u + s*v, -s*u + c*v],  s = R*e1.

    svec = R(:,1);               % s ∈ S^2
    % Smooth anchor blend (never parallel to s)
    w2 = 1 - svec(2)^2;          % = 1 - s_y^2
    w3 = 1 - svec(3)^2;          % = 1 - s_z^2
    a  = w3*[0;0;1] + w2*[0;1;0];

    u = cross(a, svec);          % ⟂ s
    u = u / norm(u);             % normalize (可用倒平方根牛顿迭代替代)
    v = cross(svec, u);          % 完成右手正交基

    cphi = dot(R(:,2), u);       % cos(phi)
    sphi = dot(R(:,2), v);       % sin(phi)
end
