function [RPM] = speedToRPM(speed, r)

% calculate rpm given linear speed (m/s) and radius r (m)

RPS = (speed)/(2*pi*r);
RPM = RPS*60;

end 
