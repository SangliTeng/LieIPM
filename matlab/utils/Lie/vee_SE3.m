function w = vee_SE3(W)
w = [W(3,2); W(1,3); W(2,1); W(1:3,4)]; %% error
end