function [speed] = accelToSpeed(a, v0, t)

% calculate speed (m/s) given acceleration a (m/s^2), initial speed v0
% (m/s), and a time interval t (s)

speed = v0 + a*t;

end 