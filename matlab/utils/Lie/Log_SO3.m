function phi = Log_SO3(R)
tn = acos((trace(R) - 1) / 2);
tn = sqrt(abs(tn)^2 + 1e-50);
phi = tn / (2 * sin(tn)) * vee_SO3((R - R'));
% phi = vee_SO3(logm(R));
end