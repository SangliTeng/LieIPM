%% ===== Plot Function =====
function plotInterpTrajWithTriads(xd, Xk, varargin)
% plotInterpTrajWithTriads
% Plot interpolated trajectory + keypoints + R triads (3 axes) in plot3.
%
% Inputs:
%   xd : 24 x K, each column [R(:); p; F(:); v]
%   Xk : 1 x M struct array with fields .R, .p
%
% Options (name-value):
%   'nTriads'    : number of triads along trajectory (default 25)
%   'triadScale' : axis length; if empty -> auto from trajectory span
%   'triadsAtKeypoints' : true/false (default true)

    p = inputParser;
    p.addParameter('nTriads', 25, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
    p.addParameter('triadScale', [], @(x)isempty(x)||(isnumeric(x)&&isscalar(x)&&x>0));
    p.addParameter('triadsAtKeypoints', true, @(x)islogical(x)&&isscalar(x));
    p.parse(varargin{:});
    nTriads = p.Results.nTriads;
    sUser   = p.Results.triadScale;
    triadsAtKeypoints = p.Results.triadsAtKeypoints;

    K = size(xd, 2);

    % unpack p(t)
    p_traj = xd(10:12, :);

    % unpack R(t)
    R_traj = zeros(3,3,K);
    for k = 1:K
        R_traj(:,:,k) = reshape(xd(1:9, k), 3, 3);
    end

    % keypoint positions
    M = numel(Xk);
    p_k = zeros(3, M);
    for i = 1:M
        p_k(:, i) = Xk(i).p(:);
    end

    % choose triad indices
    idx = unique(round(linspace(1, K, min(nTriads, K))));

    % triad scale
    if isempty(sUser)
        span = max(vecnorm(p_traj - mean(p_traj,2), 2, 1));
        if span < 1e-9, span = 1; end
        s = 0.08 * span;
    else
        s = sUser;
    end

    hold on; grid on; axis equal;
    xlabel('x'); ylabel('y'); zlabel('z');
    title('Interpolated trajectory + keypoints + R triads');

    % trajectory
    plot3(p_traj(1,:), p_traj(2,:), p_traj(3,:), 'LineWidth', 2);

    % keypoints
    plot3(p_k(1,:), p_k(2,:), p_k(3,:), 'o', 'MarkerSize', 7, 'LineWidth', 2);

    % triads along trajectory
    for ii = 1:numel(idx)
        k = idx(ii);
        drawTriad(p_traj(:,k), R_traj(:,:,k), s);
    end

    % triads at keypoints
    if triadsAtKeypoints
        for i = 1:M
            drawTriad(Xk(i).p(:), Xk(i).R, 1.3*s);
            text(Xk(i).p(1), Xk(i).p(2), Xk(i).p(3), sprintf('  x_%d', i-1));
        end
    end

    legend({'interp p(t)', 'keypoints'}, 'Location', 'best');
    view(3);
end

function drawTriad(p, R, s)
% drawTriad: draw 3 axes of rotation matrix R at position p.
% R*[1;0;0], R*[0;1;0], R*[0;0;1] are the world-frame directions of body axes.
% Equivalently: columns of R.

    ex = R * [1; 0; 0] * s;   % R * e_x
    ey = R * [0; 1; 0] * s;   % R * e_y
    ez = R * [0; 0; 1] * s;   % R * e_z

    quiver3(p(1),p(2),p(3), ex(1),ex(2),ex(3), 0,'r', 'LineWidth', 1);
    quiver3(p(1),p(2),p(3), ey(1),ey(2),ey(3), 0,'g', 'LineWidth', 1);
    quiver3(p(1),p(2),p(3), ez(1),ez(2),ez(3), 0,'b', 'LineWidth', 1);
end