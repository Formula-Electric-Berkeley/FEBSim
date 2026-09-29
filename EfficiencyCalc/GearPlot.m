function GearPlot(TorqueCurve)

%Script that plots modified motor curve vs. allowed static and dynamic
%wheel grip

% Tcurve needs to be a matlab Table
% with first column being RPM and second column being Torque
hold on
title('Wheel Torque vs Wheel RPM with Different Gear Ratios')
ylabel('Wheel Torque [Nm]')
xlabel('Wheel RPM')
GearRatios = 3:0.5:5; % configure range of gear reduction ratios
for G = GearRatios
    plot(TorqueCurve.RPM/G, TorqueCurve.Torque*G*0.9) % accounting for 90% drivetrain efficiency
end
    plot(xlim, 731.8248*[1 1]) % plot known grip-allowed wheel torque
    plot(xlim, 939.0672*[1 1]) % plot grip-allowed wheel torque WITH weight transfer
    legend([string(GearRatios),'static grip','dynamic grip'])
end