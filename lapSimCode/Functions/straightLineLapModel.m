function [output] = straightLineLapModel(m, wd, Cd, Af, e, gr, tr, Cfx, Cfy, Cr, power, tStep, tMax)

% This function takes in a set of variables to produce an matrix of time,
% speed, accel, and distance

% Daughter functions:
   % speedToRPM
   % correspTorqueFromRPM
   % distanceFromAccel
   % accelToSpeed

% m = mass of vehicle 
% wd = weight distribution (unused)
% Cd = drag coefficient
% Af = frontal area
% e = drivetrain efficiency
% gr = gear ratio
% tr = tire radius
% Cfx = coefficient of lateral friction
% Cfy = coefficient of longitudinal friction
% Cr = coefficient of rolling resistance
% power = motor rpm and torque curve (given in a [x, 2] matrix)
% tStep = time intervals between each calculated speed/acceleration
% tMax = how long we want to run the simulation for


% straight line acceleration model

% constants
g = 9.81;
ffx = Cfx*g*m;
ffy = Cfy*g*m;
frr = Cr*m/tr;

RPM = power(:, 1);
% real RPM is free speed of motor divided by gear ratio
realRPM = RPM/gr;

torque = power(:, 2);
% real torque is free torque of motor multiplied by the gear ratio and
% motor efficiency
realTorque = torque*gr*e;

% initialize variable for current speed
s = 0; 

% matrix of t, s, and a
output = zeros(1,5);

% distance
d = 0;

for t = 0:tMax/tStep
    time = t*tStep;
    currRPM = speedToRPM(s, tr);
    % match the current speed to its corresponding torque on the power
    % matrix
    currTorque = correspTorqueFromRPM(currRPM, realRPM, realTorque);
    fIdeal = currTorque/tr;
    fSlip = min(ffy, fIdeal);
    fDrag = Cd*Af*1.225*(s^2)/2;
    fReal = fSlip - fDrag-frr;
    accel = fReal/m;
    d = distanceFromAccel(d, s, tStep, accel);
    output(t+1, :) = [time, s, accel, d, currTorque];
    sNew = accelToSpeed(accel, s, tStep);
    s = sNew;
end 

end