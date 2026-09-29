function [output] = straightAccelFromSpeedMax(m, wd, Cd, Af, e, gr, tr, Cfx, Cfy, Cr, power, tStep, V0, Ts, t0, d0)

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
s = V0; 

% distance
d = 0;

t = 1;

output = [];
output1 = [];

while s < Ts
    
    time = t*tStep;
    currRPM = speedToRPM(s, tr);
    currTorque = correspTorqueFromRPM(currRPM, realRPM, realTorque);
    fIdeal = currTorque/tr;
    fSlip = min(ffy, fIdeal);
    fDrag = Cd*Af*1.225*(s^2)/2;
    fReal = fSlip - fDrag-frr;
    accel = fReal/m;
    d = distanceFromAccel(d, s, tStep, accel);
    output1(t, :) = [time+t0, s, accel, d + d0, currTorque];
    sNew = accelToSpeed(accel, s, tStep);
    s = sNew;
    t=t+1;
    
end 

len = size(output1,1);
if len ~= 1
    for i=1:(len-1)
    output(i,:) = output1(i,:);
    end

else
    output(1,:) = output1(1,:);
end

end
