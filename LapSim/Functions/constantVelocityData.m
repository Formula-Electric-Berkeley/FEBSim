function [output] = constantVelocityData(s, Td, tStep, t0, d0)

% returns a matrix of time, speed, accel, and distance given a constant
% velocity s (m/s), a target distance Td (m), and starting conditions of t0
% and d0

d = 0;

t = 1; 

output = [];
output1 = [];

while d<Td
     time = t*tStep;
     d = d + s*tStep;
     output1(t, :) = [time + t0, s, 0, d+d0, 0];
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