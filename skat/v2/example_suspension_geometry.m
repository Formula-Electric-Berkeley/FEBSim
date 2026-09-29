% Example suspension geometry generation.
% Run this file from MATLAB after adding this folder to the path.

clear;
clc;
addpath(fileparts(mfilename('fullpath')));

p = struct();
p.wheelbase = 1580;

p.front.trackWidth = 1240;
p.front.tireDiameter = 520;
p.front.scrubRadius = 18;
p.front.kpiDeg = 8.5;
p.front.casterDeg = 6.0;
p.front.rollCenterHeight = 35;
p.front.instantCenterY = -310;

p.rear.trackWidth = 1210;
p.rear.tireDiameter = 520;
p.rear.rollCenterHeight = 55;
p.rear.instantCenterY = -300;

manual = struct();
manual.frontDirectInboardLeft = [45, 255, 625];

manual.rearRockerPivotLeft = [-1490, 285, 570];
manual.rearDamperInboardLeft = [-1800, 250, 545];
manual.rearArbTubeArmEndLeft = [-1540, 115, 420];

[points, geometry, report] = generate_suspension_geometry(p, manual);

disp(points);
disp(report.summary);

% Uncomment to export the hardpoint list.
% writetable(points, 'suspension_hardpoints.csv');
