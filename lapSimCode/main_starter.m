
% Plan: eventually make this a GUI; for now just 
% m = mass of vehicle (kg)
% wd = weight distribution (%)
% Cd = drag coefficient
% Af = frontal area (m^2)
% e = drivetrain efficiency (%)
% gr = gear ratio
% tr = tire radius (m)
% Cfx = coefficient of lateral friction
% Cfy = coefficient of longitudinal friction
% Cr = coefficient of rolling resistance
% power = motor rpm and torque curve (given in a [x, 2] matrix) (rpm x Nm)
% tStep = time intervals between each calculated speed/acceleration (s)
% track = track specs (given in a [x, 2] matrix) (m x m)

m = 100;
wd = 0.5;
Cd = 10;
Af = 1;
e = 0.9;
gr = 3.3;
tr = 0.5;
Cfx = 0.1;
Cfy = 0.1;
Cr = 0.05;
power = 1;
tStep = 




trackLapModel1(m, wd, Cd, Af, e, gr, tr, Cfx, Cfy, Cr, power, tStep, track)






