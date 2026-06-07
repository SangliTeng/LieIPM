% ts = 2;
Ns = 20;
dt = 0.01;

m = 0.280;
J = diag([190.0e-6; 190.0e-6; 255.0e-6]);
% J = diag([0.1; 0.1; 0.3]) / 10;
g = [0;0;-9.81];

Jd = fun_J2Jd(J);

inertial_list = {};
for k = 1:1
    inertial_list{k}.Jd = Jd;
    inertial_list{k}.m = m;
end

B1 = 0.041010498687664041987;
B2 = 0.0051187551187551187549;
B3 = 1;

B = [-B1, -B1,  B1,  B1;
     -B1,  B1,  B1, -B1;
     -B2,  B2, -B2,  B2;
      B3,  B3,  B3,  B3];
