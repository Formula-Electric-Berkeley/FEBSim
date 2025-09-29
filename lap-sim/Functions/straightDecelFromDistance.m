function [output] = straightDecelFromDistance(m, Cd, Af, tr, Cfy, Cr, tStep, V0, Td, t0, d0)

% constants
g = 9.81;
ffy = Cfy*g*m;
frr = Cr*m/tr;

d = 0;

s = V0;

t = 1;

output1 = [];
output = [];

while d < Td
    time = t*tStep;
    fDrag = Cd*Af*1.225*(s^2)/2;
    fBrake = ffy + fDrag + frr;
    accel = -fBrake/m;
    d = distanceFromAccel(d, s, tStep, accel);
    output1(t, :) = [time + t0, s, accel, d + d0, fBrake*tr];
    sNew = accelToSpeed(accel, s, tStep);
    s = sNew;
    t = t+1;
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