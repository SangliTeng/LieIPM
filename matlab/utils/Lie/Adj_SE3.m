function AdX = Adj_SE3(R, p)
AdX = zeros(6) * p(1);
AdX(1:3, 1:3) = R;
AdX(4:6, 4:6) = R;
AdX(4:6, 1:3) = skew_SO3(p) * R;
end