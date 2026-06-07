function X = StaticInit(X0, Xg, Ns)
% initialize the system as linear interpolation between X0 and Xg
X = {X0};

% compute the total retraction
dx = zeros(12, X0.num_srb);
for k = 1:X0.num_srb
    dx(:, k) = [Log_SO3(X0.mrb(k).R' * Xg.mrb(k).R) + [0; 1; 0] * 0e-1 * (k > 0);
                Xg.mrb(k).p - X0.mrb(k).p + randn(3, 1) * 0;
                Log_SO3(X0.mrb(k).R' * Xg.mrb(k).R) / Ns;
                (Xg.mrb(k).p - X0.mrb(k).p) / Ns ] * 1;
end

for k = 2:Ns+1
    X{k} = Xg;
    % X{k} = X0.BatchRetractSO3xR3(dx(:) *  (k-1) / Ns + randn(size(dx(:))) * 0 );
end
end