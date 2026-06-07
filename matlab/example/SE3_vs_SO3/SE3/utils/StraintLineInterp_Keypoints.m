function [xd, tgrid] = StraintLineInterp_Keypoints(Xk, Tseg, dt, vFrame)

    assert(isstruct(Xk) && numel(Xk) >= 2);
    M = numel(Xk);
    assert(numel(Tseg) == M-1);
    Tseg = Tseg(:).';
    assert(all(Tseg > 0));

    for i = 1:M
        assert(all(size(Xk(i).R) == [3 3]));
        assert(numel(Xk(i).p) == 3);
    end

    if nargin < 3 || isempty(dt)
        dt = max(min(Tseg)/20, 1e-3);
    end
    if nargin < 4 || isempty(vFrame)
        vFrame = 'world';
    end

    % knot times
    tpts = zeros(1,M);
    for i = 2:M
        tpts(i) = tpts(i-1) + Tseg(i-1);
    end
    Ttot = tpts(end);

    % ------------------------------------------------------------------
    % FIXED-K grid size with guaranteed keypoint inclusion:
    % Choose N so that K = N+1 is exactly (Ttot/dt_eff)+1 with integer N,
    % and ensure all knot times tpts are on the grid by choosing dt_eff = Ttot/N
    % and requiring each tpts(i) aligns to integer multiples of dt_eff (within tol).
    % If not aligned under the user dt, we still enforce K by adjusting dt_eff.
    % ------------------------------------------------------------------
    N = round(Ttot / dt);          % number of steps
    N = max(N, 1);
    dt_eff = Ttot / N;             % adjusted dt that makes Ttot exactly N*dt_eff

    % build uniform grid of exactly K = N+1 points
    tgrid = (0:N) * dt_eff;        % 1x(K)
    tgrid(end) = Ttot;             % exact
    K = numel(tgrid);

    % ensure each knot time exists on grid (within tolerance)
    tol = 1e-10 * max(1, Ttot);
    idx_knot = zeros(1,M);
    for i = 1:M
        qi = tpts(i) / dt_eff;
        qi_round = round(qi);
        if abs(qi - qi_round) > 1e-8
            % If this triggers, your knot times cannot be represented on this grid.
            % In that case, you must either:
            % (a) accept extra points (unique([...])) OR
            % (b) choose dt so that all cumulative sums align
            % Here we FAIL loudly to preserve your "K must equal Ttot/dt + 1" requirement.
            error(['Keypoint time tpts(%d)=%.16g is not on the uniform grid. ', ...
                   'Choose dt such that all cumulative sums of Tseg align, or allow non-uniform grid.'], ...
                   i, tpts(i));
        end
        idx_knot(i) = qi_round + 1; % MATLAB index
        tgrid(idx_knot(i)) = tpts(i); % exact
    end
    % ------------------------------------------------------------------

    % positions
    P = zeros(3,M);
    for i = 1:M, P(:,i) = Xk(i).p(:); end
    h = diff(tpts); % 1x(M-1)

    % ---- C2 cubic spline second derivatives at knots (natural) ----
    M2 = zeros(3,M); % p'' at knots

    if M >= 3
        n = M-2; % number of interior knots
        d  = 2*(h(1:n) + h(2:n+1));  % length n

        if n == 1
            A = d; % 1x1
        else
            dl = h(2:n);            % length n-1
            du = h(2:n);            % length n-1
            A  = spdiags([ [dl(:);0], d(:), [0;du(:)] ], [-1,0,1], n, n);
        end

        for ax = 1:3
            y = P(ax,:).';
            rhs = 6 * ( (y(3:end) - y(2:end-1))./h(2:end).' ...
                      - (y(2:end-1) - y(1:end-2))./h(1:end-1).' );
            sol = A \ rhs;
            M2(ax,2:M-1) = sol.';
        end
    end

    % ---- evaluate spline p(t), v(t) on tgrid ----
    p_traj = zeros(3,K);
    v_world_traj = zeros(3,K);

    for k = 1:K
        t = tgrid(k);

        if t >= Ttot
            seg = M-1; tau = h(end);
        else
            seg = find(t >= tpts(1:end-1) & t <= tpts(2:end), 1, 'last');
            if isempty(seg), seg = 1; end
            tau = t - tpts(seg);
        end

        hi   = h(seg);
        yi   = P(:,seg);
        yip1 = P(:,seg+1);
        Mi   = M2(:,seg);
        Mip1 = M2(:,seg+1);

        a = (hi - tau);
        b = tau;

        p_traj(:,k) = Mi   * (a^3)/(6*hi) + Mip1 * (b^3)/(6*hi) ...
                    + (yi   - Mi  *(hi^2)/6) * (a/hi) ...
                    + (yip1 - Mip1*(hi^2)/6) * (b/hi);

        v_world_traj(:,k) = -Mi*(a^2)/(2*hi) + Mip1*(b^2)/(2*hi) ...
                          + (yip1 - yi)/hi - (Mip1 - Mi)*hi/6;
    end

    % hard guard knot positions exactly at their indices (no extra points added!)
    for i = 1:M
        p_traj(:, idx_knot(i)) = P(:,i);
    end

    % ---- rotations R(t), F(t) ----
    wR = zeros(3,M-1);
    for i = 1:(M-1)
        wR(:,i) = Log_SO3(Xk(i).R' * Xk(i+1).R);
    end

    xd = zeros(24,K);

    for k = 1:K
        t = tgrid(k);
        if t >= Ttot
            seg = M-1; s = 1.0;
        else
            seg = find(t >= tpts(1:end-1) & t <= tpts(2:end), 1, 'last');
            if isempty(seg), seg = 1; end
            s = (t - tpts(seg)) / Tseg(seg);
            s = min(max(s,0),1);
        end

        Ri = Xk(seg).R;
        Rk = Ri * Rodrigues(wR(:,seg) * s);
        Fk = eye(3); % 你如果要相对旋转就改回：Ri' * Rk

        pk = p_traj(:,k);

        switch lower(vFrame)
            case 'world'
                vk = v_world_traj(:,k) * 0;      % 不要再乘 0
            case 'body'
                vk = Rk' * v_world_traj(:,k);
            otherwise
                error('vFrame must be ''world'' or ''body''.');
        end

        xd(:,k) = [Rk(:); pk; Fk(:); vk];
    end

    % final guards
    xd(1:9,end)   = Xk(end).R(:);
    xd(10:12,end) = Xk(end).p(:);
    tgrid(end)    = Ttot;

    Rprev = Xk(end-1).R;
    Rend  = Xk(end).R;
    tmp = Rprev' * Rend;
    xd(13:21,end) = tmp(:);
end