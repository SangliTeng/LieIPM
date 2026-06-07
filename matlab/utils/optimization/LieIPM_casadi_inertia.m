function data = LieIPM_casadi_inertia(vars, y, z, s, mu, utils)
%% We test different line search and termination condition in this file.
%% Try to apply more techniques from IPOPT paper.
beta                   = utils.IPM.beta; % 0.5;
mu_decaying_rate    = utils.IPM.mu_decaying_rate; % 1 / 1.01;
% sigma_decaying_rate_sq = 0.5;
mu_decaying_rate_sq = utils.IPM.mu_decaying_rate_sq; % 1e-2;

% eps_kkt_barrier        = 1e-6;
eps_kkt_terminate      = utils.IPM.eps_kkt_terminate; % 1e-7;

barrier_kkt_acceptace_rate  = utils.IPM.barrier_kkt_acceptace_rate; % 1e-6;
barrier_cost_acceptace_rate = utils.IPM.barrier_kkt_acceptace_rate; % 1e-6;

delta   = utils.IPM.delta;
s_theta = utils.IPM.s_theta;
s_psi   = utils.IPM.s_psi;
armojo_coeff = utils.IPM.armojo_coeff;

max_line_search_count  = utils.IPM.max_line_search_count;
% eps_line_search_kkt    = 1e-4;

tau_min = utils.IPM.tau_min;

s_max = utils.IPM.s_max;

iter_max               = utils.IPM.iter_max;
% sigma = sigma;

cost_log      = [];
line_search_ct_log = [];
line_search_flag_log = [];
kkt_log       = [];
gk_log        = [];
hk_log        = [];
mu_log     = [];
t_lin_sys_log = [];
t_forloop_log = [];
barrier_kkt_violation_log = [];
overall_kkt_violation_log = [];
nnz_log = [];

ic_log = [];

flag = nan;

x = [vars.x(:); vars.u(:)];

delta_w_last = 0;
tic;
for iter = 1:iter_max

    %%
    zz = z;
    ss = s;

    A_kkt = utils.Asym_kkt(x, y, zz, ss, mu);
    A_kkt = A_kkt + A_kkt' - diag(diag(A_kkt));

    b_kkt = utils.b_kkt(x, y, zz, ss, mu);
    fk = utils.f(x, y, zz, ss, mu);
    dfk = utils.df(x, y, zz, ss, mu);

    hk = b_kkt(length(dfk)+1:length(dfk)+length(y));
    gk = b_kkt(length(dfk)+length(y)+1:length(dfk)+length(y)+length(z)) - ss;

    dhk = A_kkt(length(dfk)+1:length(dfk)+length(y), 1:length(dfk));
    dgk = A_kkt(length(dfk)+length(y)+1:length(dfk)+length(y)+length(z), 1:length(dfk));
    %%

    sd = max([s_max, (norm(y, 1) + norm(z, 1)) / (length(y) + length(z))]) / s_max;
    sc = max([s_max,  norm(z, 1) / length(z) ]) / s_max;
    
    if ~isempty(z)
        % residual of the original problem
        E = max([norm(dfk(:) + dhk' * y + dgk' * z, inf) / sd; ...
            norm(hk(:), inf); ...
            norm(z .* s - mu, inf) / sc]);

        % residual of the barrie problem
        E0 = max([norm(dfk(:) + dhk' * y + dgk' * z, inf) / sd; ...
            norm(hk(:), inf); ...
            norm(z .* s - 0, inf) / sc]);
    else
        % residual of the original problem
        E = max([norm(dfk(:) + dhk' * y, inf) / sd; ...
            norm(hk(:), inf)]);

        % residual of the barrie problem
        E0 = E;
    end

    barrier_kkt_violation_log = [barrier_kkt_violation_log; E];
    overall_kkt_violation_log = [overall_kkt_violation_log; E0];

    % E0
    if E0 < eps_kkt_terminate% norm(b_kkt, inf) < 1e-3
        flag = 1;
        % norm(b_kkt, inf)
        % E0
        break;
    end
    
    % disp([E, E0])
    % save("ipm_test.mat")

    if E < 10 * mu % eps_kkt_barrier % norm(r) < eps_kkt_vec
        % sigma = min([sigma_decaying_rate * sigma, sigma_decaying_rate_sq * norm(b_kkt)^2]);
        % KKT paper, converge slower...

        % IPOPT paper, linear v.s, super-linear, then cut to threshold
        mu = min([mu_decaying_rate * mu, mu_decaying_rate_sq * mu^1.999]);
        mu = max([mu, eps_kkt_terminate / 10]);
        % "update central path"
    end

    nnz_log = [nnz_log; nnz(A_kkt)];
    t1 = toc;
    
    b_kkt_new = [b_kkt(1:end-length(s))];
    b_kkt_new(end-length(s)+1:end) = b_kkt_new(end-length(s)+1:end) - b_kkt(end-length(s)+1:end) ./ z;
    
    %% inertial correction starts here... 
    
    delta_bar_w_min = 1e-20;
    delta_bar_w_0   = 1e-4; % 1e-4;
    delta_bar_w_max = 1e40;
    k_bar_w_plus    = 100;
    k_w_plus        = 8;
    k_w_minus       = 1 / 3; % 1 / 3;
    k_c             = 1 / 4;
    delta_bar_c     = 1e-8;

    delta_c = 0;
    
    [L, D, P] = ldl(A_kkt); % , 'vector');
    inertia = count_inertia(D);

    nx = length(dfk);
    ny = size(A_kkt, 1) - nx;
    
    ic_required = inertia(2) ~= ny || inertia(3)~= 0; % (inertia(1) ~= nx || inertia(2) ~= ny || inertia(3)~= 0);

    %% Apr. 21 debug
    % [S_, V_, D_] = svd(full(A_kkt));
    % s_ = diag(S_)
    % evs = eig(A_kkt);

    %%
    
    if ic_required
        % IC-2
        if inertia(3)~= 0
            delta_c = delta_bar_c * mu^k_c;
        end
    
        % IC-3
        if delta_w_last == 0
            delta_w = delta_bar_w_0;
        else
            delta_w = max([delta_bar_w_min, k_w_minus * delta_w_last]);
        end
    end
    
    ic_count = 0;
    while (ic_required)
        ic_count = ic_count + 1;
        % if (inertia(1) ~= length(dfk) || inertia(2) ~= length(dh) || inertia(3)~= 0)
            % ct_inertia = ct_inertia + 1;

        % disp("Inertia correction required!")

        % IC-4: solve the modified system
        A_kkt_c = A_kkt;
        blk1 = 1:length(dfk);
        A_kkt_c(blk1, blk1) = A_kkt_c(blk1, blk1) + delta_w * eye(length(dfk));
        blk2 = length(dfk)+1:length(dfk)+length(hk);
        A_kkt_c(blk2, blk2) = A_kkt_c(blk2, blk2) - delta_c * eye(length(hk));
        
        [L, D, P] = ldl(A_kkt_c); % , 'vector');
        
        inertia = count_inertia(D);

        ic_required = inertia(2) ~= ny; % (inertia(1) ~= nx || inertia(2) ~= ny || inertia(3)~= 0);
        
        if ~ic_required
            disp("Inertia correction succeed!")
            delta_w_last = delta_w;
        else
            % disp("Inertia correction failed!")
            % IC-5
            if delta_w_last == 0
                delta_w = k_bar_w_plus * delta_w;
            else
                delta_w = k_w_plus * delta_w;
            end            
        end
        
        if delta_w > delta_bar_w_max % || ic_count > 2
            disp("[FUCK] Inertia correction failed!")
            break;
        end
    end
    sol = -P*(L'\(D\(L\(P'*b_kkt_new))));
    sol_k = [sol; 
        z.^-1 .* (- b_kkt(end-length(s)+1:end) - s.* sol(end-length(s)+1:end))];
    t2 = toc;
    t_lin_sys_log(end+1) = t2 - t1;
    ic_log(end + 1) = ic_count;
    %%

    if sum(isnan(sol_k)) > 0 || norm(sol_k, 'inf') > 1e8
        flag = 0;
        "NaN detected!"
        break;
    end

    %% Fraction to boundary rule ...
    dx   = sol_k(1:length(dfk));
    dy = sol_k(length(dfk)+1:length(dfk)+length(y));
    dz  = sol_k(end-length(s)-length(z)+1:end-length(s));
    ds   = sol_k(end-length(s)+1:end);
    tau  = max([tau_min, 1 - mu]);
    % sgn_alpha  = sign(ds);
    alpha_s_list   = (-tau * s) ./ ds;
    alpha_s_max    = min([alpha_s_list(ds<0); 1]); % alpha_s_min  = max([alpha_s_list(ds>0); 0]);

    alpha_z_list  = (-tau * z) ./ dz; % alpha_mu_min = max([alpha_mu_list(dmu>0); 0]);
    alpha_z_max   = min([alpha_z_list(dz<0); 1]);

    %% Filter-based method + Armijo condition
    alpha  = 1;
    % r0 = kkt_vector_field(X_new, U_new, lam_new, mu_new, s_new, sigma, utils);
    r0     = norm(hk(:), 1) + sum(gk(gk > 0));
    r0_min = 1e-4 * max([1, r0]);
    % r02 = ;
    flag_ls = nan;
    ct = 1;
    while true
        sol_exec = [dx * alpha;
            dy;
            alpha_z_max * dz;
            alpha_s_max  * ds];

        [x_new, y_new, z_new, s_new] = Retraction_xyz_mex(x, y, z, s, sol_exec);

        f_trial = utils.f(x_new, y_new, z_new, s_new, mu);

        [hk_trial, gk_trial]   =  utils.hg(x_new, y_new, z_new, s_new, mu);

        % constraint violation
        r = norm(hk_trial(:), 1) + sum(gk_trial(gk_trial > 0));

        % switching condition, whether filter or Armijo condition
        dfk_trial = utils.df(x_new, y_new, z_new, s_new, mu); dfk_trial = dfk_trial(:);

        progress = dfk_trial' * dx;
        
        flag1 = progress < 0;

        if flag1
            flag2 = alpha * (-progress)^s_psi > delta * r^s_theta;
        else
            flag2 = false;
        end
        flag3 = r <= r0_min;

        % disp([flag1, flag2, flag3])
        % fprintf('%.10f, %.10f, %.10f \n', f_trial, r, progress);
        % disp()

        if flag1 && flag2 && flag3% if constraints violation is small enough, use Armijo condition to obtain sufficie

            % if flag1 && flag2
            % f_trial = fun_f(X_new, U_new, utils);
            if f_trial - mu' * sum(log(s_new)) < fk - mu' * sum(log(s)) + armojo_coeff * alpha * progress
                disp("Armijo condition satisfied at iter: " + num2str(ct))
                flag_ls = 1;
                break;
            end
        else
            % violation of constraints
            flag1 = norm(r) < (1 - barrier_kkt_acceptace_rate) * norm(r0);
            % sufficient decreasing of barrier objective
            flag2 = f_trial - mu' * sum(log(s_new)) < fk - mu' * sum(log(s)) - barrier_cost_acceptace_rate * norm(r0);
            % maximal number of line search
            % flag3 = ct > max_line_search_count;
            if flag1 || flag2 %  || flag3
                flag_ls = 2;
                disp("eq(18) satisfied at iter: " + num2str(ct))
                break
            end
        end
        if ct > max_line_search_count
            flag_ls = -1;
            disp("line-search maximal iter reached at iter: " + num2str(ct))
            break
        end
        alpha = alpha * beta;
        ct = ct + 1;
    end
    % disp(ct)
    line_search_ct_log = [line_search_ct_log; ct];
    line_search_flag_log = [line_search_flag_log; flag_ls];

    %%

    % Update states
    x    = x_new;
    y    = y_new(:);
    z    = z_new(:);
    s    = s_new(:);

    cost_log  = [cost_log, fk];
    mu_log = [mu_log; mu];

    flag = 0;

    t_forloop_log = [t_forloop_log; toc];
    % if iter == 9
    %     "j"
    % end
end
tend = toc;
t_forloop_log = [t_forloop_log; tend];
data.t_total = tend;

data.xu  = x;
% data.u  = u;
data.s     = s;
data.mu    = z;
data.lam   = y;
data.sigma = mu;

data.cost_log  = cost_log;
data.kkt_log   = kkt_log;
data.gk_log    = gk; % only the last round at this time
data.hk_log    = hk; % only the last round at this time
data.sigma_log = mu_log;
data.barrier_kkt_violation_log = barrier_kkt_violation_log;
data.overall_kkt_violation_log = overall_kkt_violation_log;

data.terminate_flag = flag;
data.t_lin_sys_log  = t_lin_sys_log;
data.t_forloop_log = t_forloop_log;

data.line_search_ct_log = line_search_ct_log;
data.line_search_flag_log = line_search_flag_log;

data.nnz_log = nnz_log;

data.ic_log  = ic_log;

data.iter = iter;
data.flag = flag;

end

% data.gk_log    = gk_log;
% data.hk_log    = hk_log;

% kkt_log   = [kkt_log, kkt_vector_field(X_new, U_new, y_new, z_new, s_new, 0, utils)];
% gk_log    = [gk_log, gk];
% hk_log    = [hk_log, hk];

% data.t1_lin_sys_log  = t1_lin_sys_log;
% data.t2_lin_sys_log  = t2_lin_sys_log;

function inertia = count_inertia(D)
    % D is block diagonal: 1x1 or 2x2 blocks
    n = size(D,1);
    i = 1;
    pos = 0; neg = 0; zero = 0;
    
    disp("Inertia correction triggered!")
    while i <= n
        if i < n && D(i,i+1) ~= 0  % 2x2 block
            B = D(i:i+1,i:i+1);
            eigvals = eig(B);
            pos = pos + sum(eigvals > 1e-12);
            neg = neg + sum(eigvals < -1e-12);
            zero = zero + sum(abs(eigvals) <= 1e-12);
            i = i + 2;
        else  % 1x1 block
            d = D(i,i);
            if d > 1e-12
                pos = pos + 1;
            elseif d < -1e-12
                neg = neg + 1;
            else
                zero = zero + 1;
            end
            i = i + 1;
        end
    end

    inertia = [pos, neg, zero];
end