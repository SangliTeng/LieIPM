clear pendulum X0 Xg;
num_trial = 100;
% rng(4);
for k = 1:num_trial
    % Before RSS
    % ax0 = [randn; randn; randn]; ax0 = ax0 / norm(ax0) * pi * 1 * rand; 

    % Before Rebuttal
    ax0 = [randn; randn; 0]; ax0 = ax0 / norm(ax0) * pi * 0.999;
    % ax0 = [0; 0; 0]; % ax0 = ax0 / norm(ax0) * 0;

    pendulum(k) = RigidBody3D(1, m, J);
    pendulum(k).R = Rodrigues(ax0);
    pendulum(k).p = [(rand - .5) * 1; (rand - .5) * 1; rand * 1]; % [0; 0; 0];
    % pendulum(k).p = [0; 0; 0]; % [0; 0; 0];

    pendulum(k).F = Rodrigues((rand(3, 1) - 0.5) * dt * 1); % [0.0, 0.0, 0.0]
    pendulum(k).v = (rand(3, 1) - 0.5) * 1;

    X0_all(k) = MultiRigidBody3D(1, pendulum(k), dt);

    pk = pendulum(k);
    % pk.p = (rand(3, 1) - 0.5) * 4;
    pk.p = [0; 0.5; 1.25];

    pk.R = eye(3);
    pk.F = eye(3);
    pk.v = zeros(3, 1);

    Xg_all(k) = MultiRigidBody3D(1, pk, dt);
end