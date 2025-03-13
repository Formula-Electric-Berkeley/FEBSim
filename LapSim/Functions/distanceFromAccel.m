function [distance] = distanceFromAccel(x0, v, t, a)

% calculates distance traveled based on initial position x0 (m), initial
% velocity v (m/s) over a period time t (s) with an acceleration of a
% (m/s^2)

distance = x0 + v*t + (1/2)*a*t^2;

end