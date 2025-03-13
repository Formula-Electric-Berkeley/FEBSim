function [torque] = correspTorqueFromRPM(rpm, rpmList, torqueList)

% uses indexing to return the corresponding torque from the closest RPM
% value to the input RPM

len = length(rpmList);

index = 1; 
minDiff = 10000;

for i = 1:len
    diff = abs(rpmList(i) - rpm);
    if diff < minDiff
        minDiff = diff;
        index = i;
    end
end

torque = torqueList(index);

end