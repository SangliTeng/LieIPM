clear;clc;close all;
% rng(1)% rng(103 + 1)
%% A kinematics system
warning('off')

load_system_constant_122_new
% load_init_state
% load_controller_param_trace
% gen_test_cases;
gen_init_guess_hiper;
%%/root/Leaning-Hybrid-Systems/Toy_Example/data/bouncing_balls_100balls_1000bounces_abs0.0_20250726_212807.mat
data_log = {};

dt_schedule = [0.01 * ones(10, 1);
    linspace(0.01, 0.2, 10)'];

% dt_schedule = 0.02 * ones(20, 1);

t_schedule = 0;
for k = 2:length(dt_schedule)
    t_schedule(k) = t_schedule(k-1) + dt_schedule(k);
end


% 0.25  * ones(10, 1)];

% dt_schedule = 0.15 * ones(40, 1);
% rng(68) % 33 iters to converge
% rng(21) % 52 iter
% rng(9191) % 52 iter

parfor k = 1:num_trial
    %%
    k
    % load_init_state;
    utils = struct();
    num_srb = 1;

    X0 = X0_all(k);
    utils.X0 = X0;

    Xg = Xg_all(k);

    utils.Xg = Xg;
    Xg_init = Xg;
    %%

    %% set the controller constants
    idx0 = 1:12;
    utils.initial_state_idx = idx0;
    for kk = 1:num_srb-1
        utils.initial_state_idx = [utils.initial_state_idx, idx0 + 12 * (kk)];
    end

    % utils.Nu  = 4; %
    % utils.Nx  = 12 * num_srb;
    utils.Nx0 = 12;
    % utils.Ns  = Ns;
    utils.Nh  = 12 * num_srb;
    utils.Ng  = 8 + 0;
    utils.dt  = dt;

    %% set the weighting matrices
    % scale = 1e-1; % scaling matters for better convergence... 
    % 
    % QR = diag([1, 1, 1]) * 10 * scale;
    % QF = diag([1, 1, 1]) * 10 * scale;
    % Qp = diag([1, 1, 1]) * 10 * scale;
    % Qv = diag([1, 1, 1]) * 10 * scale;
    % 
    % 
    % % W = diag([1, 1, 1, 1]) * 10;
    % W = diag([1, 1, 1, 1]) * .01 * scale;
    % 
    % PR = QR * 100 * scale;% diag([1, 1, 1]) * 100 * 10;
    % PF = QF * 100 * scale;% diag([1, 1, 1]) * 100 * 10;
    % Pp = Qp * 100 * scale;% diag([1, 1, 50])* 100;
    % Pv = Qv * 100 * scale;% diag([1, 1, 50]) * 100;

    scale = 1e-2;

    % stable version
    % QR = diag([1, 1, 1]) * 10 * scale;
    % QF = diag([1, 1, 1]) * 1000 * scale;
    % Qp = diag([1, 1, 1]) * 50 * scale;
    % Qv = diag([1, 1, 1]) * 50 * scale;
    
    QR = diag([1, 1, 1]) * 10 * scale;
    QF = diag([1, 1, 1]) * 1000 * scale;
    Qp = diag([1, 1, 1]) * 100 * scale;
    Qv = diag([1, 1, 1]) * 50 * scale;

    PR = QR * 100 * scale;% diag([1, 1, 1]) * 100 * 10;
    PF = QF * 100 * scale;% diag([1, 1, 1]) * 100 * 10;
    Pp = Qp * 100 * scale;% diag([1, 1, 50])* 100;
    Pv = Qv * 100 * scale;% diag([1, 1, 50]) * 100;

    
    
    % W = diag([1, 1, 1, 1]) * 10;
    W = diag([1, 1, 1, 1]) * .01 * scale;

    utils.input_weights = W;

    for kk = 1:1
        utils.stage_state_weights{kk}    = [QR(:); Qp(:); QF(:); Qv(:)];
        utils.terminal_state_weights{kk} = [PR(:); Pp(:); PF(:); Pv(:)];
    end

    utils.Xg = Xg;
    %% assign the util functions
    IPM = struct();
    IPM.beta = 0.5;
    IPM.mu_decaying_rate = 1 / 1.01;
    IPM.mu_decaying_rate_sq = 1e-2;
    IPM.eps_kkt_terminate = 1e-12; % 1e-7;
    IPM.barrier_kkt_acceptace_rate = 1e-6;
    % IPM.barrier_kkt_acceptace_rate = 1e-6;

    IPM.delta = 1;
    IPM.s_theta = 1.1;
    IPM.s_psi = 2.3;
    IPM.armojo_coeff = 1e-4;

    IPM.max_line_search_count  = 20;

    IPM.tau_min = 0.995;

    IPM.s_max = 100;
    IPM.iter_max = 100;

    utils.IPM = IPM;

    %%
    X_op = StraightLineInitDrone(X0, Xg_init, Ns);
    U_op = {};
    for kk = 1:Ns+1
        if kk <= Ns
            U_op{kk}   = [randn(3, 1) * 0.0; 9.81 * .1] * 0 + 0.68 * 0;
        end
    end
    y     = randn(Ns * utils.Nh + utils.Nx0, 1) * 0.01; % dual variable
    s     =  rand(Ns * utils.Ng, 1) * 0 + 1;
    z     =  rand(Ns * utils.Ng, 1) * 0 + 1;
    sigma = 1e-1;

    if utils.Ng == 0
        s = [];
        z = [];
    end

    vars  = struct();
    vars.U_op = U_op;
    vars.X_op = X_op;

    %%

    x = [];
    xd = [];
    for kk = 1:length(X_op)
        xk = [X_op{kk}.mrb(1).R(:);
            X_op{kk}.mrb(1).p(:);
            X_op{kk}.mrb(1).F(:);
            X_op{kk}.mrb(1).v(:)];

        xdk = [utils.Xg.mrb(1).R(:);
            utils.Xg.mrb(1).p(:);
            utils.Xg.mrb(1).F(:);
            utils.Xg.mrb(1).v(:)];

        % xdk = [vars.X_op{kk}.mrb(1).R(:);
        %     vars.X_op{kk}.mrb(1).p(:);
        %     vars.X_op{kk}.mrb(1).F(:);
        %     vars.X_op{kk}.mrb(1).v(:)];
        x = [x, xk];
        xd = [xd, xdk];
    end

    vars.x = x;
    vars.u = cell2mat(U_op);
    utils.xd = xd;
    %%

    x0 = [utils.X0.mrb(1).R(:);
        utils.X0.mrb(1).p(:);
        utils.X0.mrb(1).F(:);
        utils.X0.mrb(1).v(:)];
    
    u_offset = 0.;
    utils.A_kkt = @(xx, yy, zz, ss, mu)A_kkt_mex(xx, yy, zz, ss, mu,...
        x0,...
        utils.stage_state_weights{1},...
        utils.terminal_state_weights{1},...
        utils.input_weights(:),...
        xd(:),...
        vars.u(:) * 0 + u_offset,...
        dt_schedule);

    utils.Asym_kkt = @(xx, yy, zz, ss, mu)Asym_kkt_mex(xx, yy, zz, ss, mu,...
        x0,...
        utils.stage_state_weights{1},...
        utils.terminal_state_weights{1},...
        utils.input_weights(:),...
        xd(:),...
        vars.u(:) * 0 + u_offset,...
        dt_schedule);

    utils.b_kkt = @(xx, yy, zz, ss, mu)b_kkt_mex(xx, yy, zz, ss, mu,...
        x0,...
        utils.stage_state_weights{1},...
        utils.terminal_state_weights{1},...
        utils.input_weights(:),...
        xd(:),...
        vars.u(:) * 0 + u_offset,...
        dt_schedule);

    utils.f = @(xx, yy, zz, ss, mu)f_mex(xx, yy, zz, ss, mu,...
        x0,...
        utils.stage_state_weights{1},...
        utils.terminal_state_weights{1},...
        utils.input_weights(:),...
        xd(:),...
        vars.u(:) * 0 + u_offset,...
        dt_schedule);

    utils.hg = @(xx, yy, zz, ss, mu)hg_mex(xx, yy, zz, ss, mu,...
        x0,...
        utils.stage_state_weights{1},...
        utils.terminal_state_weights{1},...
        utils.input_weights(:),...
        xd(:),...
        vars.u(:) * 0 + u_offset,...
        dt_schedule);

    utils.df = @(xx, yy, zz, ss, mu)df_mex(xx, yy, zz, ss, mu,...
        x0,...
        utils.stage_state_weights{1},...
        utils.terminal_state_weights{1},...
        utils.input_weights(:),...
        xd(:),...
        vars.u(:) * 0 + u_offset,...
        dt_schedule);

    utils.hg = @(xx, yy, zz, ss, mu)hg_mex(xx, yy, zz, ss, mu,...
        x0,...
        utils.stage_state_weights{1},...
        utils.terminal_state_weights{1},...
        utils.input_weights(:),...
        xd(:),...
        vars.u(:) * 0 + u_offset,...
        dt_schedule);

    %%
    tic
    % data_log{k} = LieIPM_casadi(vars, y, z, s, sigma, utilss
    data_log{k} = LieIPM_casadi_inertia(vars, y, z, s, sigma, utils);
    % data_log{k} = LieIPM_casadi_filter(vars, y, z, s, sigma, utils);
    toc
end
%% generate some real data for debug

% cpp_data.x = data_log{1}.xu;
% cpp_data.y = data_log{1}.lam;
% cpp_data.z = data_log{1}.mu;
% cpp_data.s = data_log{1}.s;
% cpp_data.mu = data_log{1}.sigma;
% cpp_data.x0 = x0;
% cpp_data.stage_state_weights = utils.stage_state_weights{1};
% cpp_data.terminal_state_weights = utils.terminal_state_weights{1};
% cpp_data.input_weights = utils.input_weights(:);
% cpp_data.xd = xd(:);
% cpp_data.u  = vars.u(:) * 0;
% cpp_data.dt  = dt_schedule;
% save('ipm_warm.mat', '-struct', 'cpp_data')
%%
close all
figure()
hold on
box on
grid on
for k = 1:num_trial
    % data_log{k}.t_total
    if length(data_log{k}.overall_kkt_violation_log) < 60
        plot(data_log{k}.overall_kkt_violation_log, '--')
    end
end
plot(1:60, (1:60) * 0 + 1e-12, 'k--', LineWidth=1)
set(gca, 'YScale', 'log')
xlabel('Iterations', 'Interpreter','latex', 'FontSize', 15)
ylabel('KKT Violations', 'Interpreter','latex', 'FontSize', 15)
title('Convergence of Lie-IPM on 1000 random cases', 'Interpreter','latex', 'FontSize', 15)
%%
t_log = [];
f_log = [];
theta_log = [];
t_lin_log = [];
t_lin_log_all = [];
for k = 1:num_trial
    % if data_log{k}.t_total < 1
    t_log(k) = data_log{k}.t_total;
    t_lin_log(k) = sum(data_log{k}.t_lin_sys_log);
    t_lin_log_all = [t_lin_log_all; data_log{k}.t_lin_sys_log'];
    f_log(k) = data_log{k}.flag;

    theta_log = [theta_log,norm(Log_SO3(pendulum(k).R))];
end
%%
% plot(sort(t_log))
%%
figure()
% close
ct_success = 0;

for id = 1:length(data_log) / 100
    % id = randi(num_trial);
    if data_log{id}.flag
        ct_success = ct_success + 1;
        xlog = data_log{id}.xu(1:24 * (Ns + 1) );
        xlog = reshape(xlog, [24, Ns + 1]);

        ulog = data_log{id}.xu(24 * (Ns + 1)+1:end);
        ulog = reshape(ulog, [4, Ns]);


        eR = [];
        for k = 1:Ns + 1
            Rk = xlog(1:9, k);
            Rk = reshape(Rk, [3, 3]);
            % eRk = Rk - eye(3);
            % eRk = trace(eRk' * eRk);
            eRk = vee_SO3(logm(Rk));
            eR = [eR; norm(eRk)];
        end

        subplot(3, 2, 1)
        hold on
        plot3(xlog(10, :), xlog(11,:), xlog(12,:), '-o')
        box on
        grid on

        daspect([1, 1, 1])


        subplot(3, 2, 2)
        hold on
        plot([0,t_schedule], eR, '-o')
        box on
        grid on
    
        for j = 1:4
            subplot(3, 2, 2 + j)
            hold on
            plot(t_schedule ,ulog(j, :), '-.')
            box on
            grid on
        end
    end
end

% disp("success number = " + num2str(ct_success))
%%
ct_success = 0;
for id = 1:length(data_log)
    if data_log{id}.flag
        ct_success = ct_success + 1;
    end
    xlog = data_log{id}.xu(1:24 * (Ns + 1) );
    xlog = reshape(xlog, [24, Ns + 1]);
    
    ulog = data_log{id}.xu(24 * (Ns + 1)+1:end);
    ulog = reshape(ulog, [4, Ns]);

    eR = [];
    for k = 1:Ns + 1
        Rk = xlog(1:9, k);
        Rk = reshape(Rk, [3, 3]);
        % eRk = Rk - eye(3);
        % eRk = trace(eRk' * eRk);
        eRk = vee_SO3(logm(Rk));
        eR = [eR; norm(eRk)];
    end
    hold on
    plot([0,t_schedule], eR, '-')
    grid on
    box on
end
ct_success
%%
xlabel("Time (s)", "FontSize", 18, "Interpreter", "latex")
ylabel("Inclication Angle", "FontSize", 18, "Interpreter", "latex")
%%
ic_log = [];
for id = 1:length(data_log)
    hold on
    ic_log(id) = sum(data_log{id}.ic_log);
end
plot(sort(ic_log), '-o')