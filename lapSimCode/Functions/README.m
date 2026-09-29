% READ ME

% This folder contains all the necessary functions to run trackLapModel and
% straightLineLapModel


% DOCUMENTATION FOR straightLineLapModel
% straightLineLapModel takes in a set of variables to calculate speed,
% acceleration, and distance traveled at every time interval tStep up to a
% certain time tMax. 



% DOCUMENTATION FOR trackLapModel
% trackLapModel takes in a similar set of variables to straightLineModel
% with a few extra. 
% this function is the mother function for all other functions I use to
% calculate speed, accel, and distance traveled at every time interval
% tStep
% this function uses 3 if statements to check for 3 different cases, at
% which it will then run different code in order to produce corresponding
% matricies that eventually get added to the mother matrix. 
% each of its child functions (straightDecelFromDistance,
% straightAccelFromDistance, speedTimeFromDistance, constantVelocityData)
% all take in similar variables as trackLapModel in order to produce their
% own outputs 