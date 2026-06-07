%% initialization for a doubele-spherical pendulum
num_srb = 1;
pendulum(1) = RigidBody3D(1, m, J);

ax0 = [randn; randn; randn]; ax0 = ax0 / norm(ax0) * pi * 1;
pendulum(1).R = Rodrigues(ax0);
pendulum(1).p = [0; 0; 0] + ([rand - .5; rand - .5; rand] * 4);
pendulum(1).F = Rodrigues(randn(3, 1) * 0.); % [0.0, 0.0, 0.0]
pendulum(1).v = randn(3, 1) * 0.;


X0 = MultiRigidBody3D(1, pendulum, dt);
utils.X0 = X0;
%% set the goal state for the spherical pendulum
Xg = MultiRigidBody3D(1, pendulum, dt);

axall = [1; 1; 1];
ax_scaling = 0.0;
ax1 = axall; ax1 = ax1 / norm(ax1) * pi * ax_scaling;

Xg.mrb(1).R = Rodrigues(ax1);
Xg.mrb(1).p = [randn; randn; rand] * 0; 

utils.Xg = Xg;
%%
Xg_init = Xg;

% axall = [1; 0; 0] * 0;
% ax_scaling = .99;
% ax1 = [0; 0; 0];

% Xg_init.mrb(1).R = Rodrigues(ax1);
% Xg_init.mrb(1).p = Xg.mrb(1).p;

% Xg_init.mrb(1).R = pendulum(1).R;
% Xg_init.mrb(1).p = pendulum(1).p;
% 
% 
utils.Xg_init = Xg_init;