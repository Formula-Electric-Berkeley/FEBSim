function [output] = trackLapModel1(m, wd, Cd, Af, e, gr, tr, Cfx, Cfy, Cr, power, tStep, track)

% m = mass of vehicle (kg)
% wd = weight distribution (%)
% Cd = drag coefficient
% Af = frontal area (m^2)
% e = drivetrain efficiency (%)
% gr = gear ratio
% tr = tire radius (m)
% Cfx = coefficient of lateral friction
% Cfy = coefficient of longitudinal friction
% Cr = coefficient of rolling resistance
% power = motor rpm and torque curve (given in a [x, 2] matrix) (rpm x Nm)
% tStep = time intervals between each calculated speed/acceleration (s)
% track = track specs (given in a [x, 2] matrix) (m x m)

% constants
g = 9.81;
ffx = Cfx*g*m;
ffy = Cfy*g*m;
frr = Cr*m/tr;
iterations = 6;


maxSpeed = zeros(1, 1);

% make 1st matrix of just radii
radii = track(:, 2);

% make 2nd matrix of just section lengths
secLength = track(:, 1);


RPM = power(:, 1);
% real RPM is free speed of motor divided by gear ratio
realRPM = RPM/gr;

torque = power(:, 2);
% real torque is free torque of motor multiplied by the gear ratio and
% motor efficiency
realTorque = torque*gr*e;

V0 = 0;

% determine maximum speeds for each turn (and straight)
for i = 1:length(radii)
    % case 1: if curved segment, determine maximum velocity to maintain traction
    % given max lateral friction
    currMaxSpeed = maxSpeedFromRadius(radii(i), ffx, m);
    % case 2: if straight segment, accelerate starting from max speed of
    % last segment
    if currMaxSpeed == 0 
        % if first segment is straight, set last max speed = 0, otherwise
        % use last input in list
        if i ~= 1
            V0 = maxSpeed(i-1);
        end
        straightLineAccel = speedTimeFromDistance(m, wd, Cd, Af, e, gr, tr, Cfx, Cfy, Cr, power, tStep, V0, secLength(i));
        currMaxSpeed = straightLineAccel(end, 2);
    end
    
    % make 3rd matrix of just max speeds
    maxSpeed(i) = currMaxSpeed;
end

% initialize 1x5 matrix for final output
output = zeros(1, 5);

t0 = 0;
d0 = 0;

for i = 1:(length(radii)-1)
    % case 1: straight into turn, need to decel
    if radii(i) == 0 
        disp('case1');
        initialLength = secLength(i);
        newLength = initialLength;
        V0 = 0;
        if i ~= 1
            V0 = maxSpeed(i-1);
        end
        % loop (iteration) times, this constant can be increased to make it
        % more accurate 
        
        % if iterations is odd, speed will be slightly underestimated, if
        % it is even, it will be slightly overestimated
        for j = 1:iterations
            currData = speedTimeFromDistance(m, wd, Cd, Af, e, gr, tr, Cfx, Cfy, Cr, power, tStep, V0, newLength);
            s = currData(end, 2);
            maxBForce = ffy;
            % slightly overestimated
            maxDecel = maxBForce/m;
            % slightly overestimated 
            timeDecel = (s - maxSpeed(i+1))/maxDecel;
            % slightly underestimated
            distanceDecel = s*timeDecel - (1/2)*maxDecel*(timeDecel)^2;
            % slightly over estimated 
            newLength = initialLength - distanceDecel;
        end
        % add accel data for newLength
        t0 = output(end, 1);
        d0 = output(end, 4);
        output = cat(1, output, straightAccelFromDistance(m, wd, Cd, Af, e, gr, tr, Cfx, Cfy, Cr, power, tStep, V0, newLength, t0, d0));
        t0 = output(end, 1);
        d0 = output(end, 4);
        % add decel data for distanceDecel
        output = cat(1, output, straightDecelFromDistance(m, Cd, Af, tr, Cfy, Cr, tStep, s, distanceDecel, t0, d0));
        t0 = output(end, 1);
        d0 = output(end, 4);
    end
    
    % case 3, turn into turn, need to decel
    if radii(i) ~= 0 && maxSpeed(i)>maxSpeed(i+1)
        % case 3a
        if maxSpeed(i-1)>=maxSpeed(i)  
            disp('case3a');
            % constant velocity calculations
            initialLength = secLength(i);
            s = maxSpeed(i);
            % braking force = friction + 1/2 drag force in order to have a
            % better estimate
            maxBForce = ffy + (Cd*Af*1.225*(s^2)/2)/2;
            maxDecel = maxBForce/m;
            timeDecel = (s - maxSpeed(i+1))/maxDecel;
            distanceDecel = s*timeDecel - (1/2)*maxDecel*(timeDecel)^2;
            newLength = initialLength - distanceDecel;
            % add constant velocity data to output
            output = cat(1, output, constantVelocityData(s, newLength, tStep, t0, d0));
            t0 = output(end, 1);
            d0 = output(end, 4);
            % add decel data to output
            output = cat(1, output, straightDecelFromDistance(m, Cd, Af, tr, Cfy, Cr, tStep, s, distanceDecel, t0, d0));
            t0 = output(end, 1);
            d0 = output(end, 4);
        end
        
        % case 3b
        if maxSpeed(i-1)<maxSpeed(i)
            disp('case3b');
            initialLength = secLength(i);
            s = output(end, 2);
            distanceDecel = 0; 
            distanceLeft = initialLength;
            t=1;
            d=0;
            while (distanceDecel < abs(distanceLeft)) && (d < initialLength)
                if s < maxSpeed(i)
                    time = t*tStep;
                    currRPM = speedToRPM(s, tr);
                    currTorque = correspTorqueFromRPM(currRPM, realRPM, realTorque);
                    fIdeal = currTorque/tr;
                    fSlip = min(ffy, fIdeal);
                    fDrag = Cd*Af*1.225*(s^2)/2;
                    fReal = fSlip - fDrag-frr;
                    accel = fReal/m;
                    disp('s');
                    disp(s);
                    d = distanceFromAccel(d, s, tStep, accel);
                    distanceLeft = initialLength - d;
                    disp('d');
                    disp(d);
                    disp('distanceLeft');
                    disp(distanceLeft);
                    maxBForce = ffy + Cd*Af*1.225*(s^2)/2;
                    maxDecel = maxBForce/m;
                    if s > maxSpeed(i+1)
                        timeDecel = abs((s - maxSpeed(i+1))/maxDecel);
                        disp('timeDecel');
                        disp(timeDecel);
                        distanceDecel = s*timeDecel - (1/2)*maxDecel*(timeDecel)^2;
                    else
                        distanceDecel = 0;
                    end
                    disp('distanceDecel');
                    disp(distanceDecel);
                    output = cat(1,output,[time+t0, s, accel, d + d0, currTorque]);
                    sNew = accelToSpeed(accel, s, tStep);
                    s = sNew;
                    t=t+1;
                end
                if s >= maxSpeed(i)
                    time = t*tStep;
                    d = d+s*tStep;
                    maxBForce = ffy + (Cd*Af*1.225*(s^2)/2)/2;
                    maxDecel = maxBForce/m;
                    timeDecel = (s - maxSpeed(i+1))/maxDecel;
                    distanceDecel = s*timeDecel - (1/2)*maxDecel*(timeDecel)^2;
                    distanceLeft = initialLength - d;
                    output = cat(1, output, [time+t0, s, 0, d + d0, currTorque]);
                    t=t+1;
                end
            end
            t0 = output(end, 1);
            d0 = output(end, 4);
            output = cat(1, output, straightDecelFromDistance(m, Cd, Af, tr, Cfy, Cr, tStep, s, distanceDecel, t0, d0));
            t0 = output(end, 1);
            d0 = output(end, 4);
        end
        
    end
    
    
    % case 2, turn into turn, need to accel
    % case 4, turn into straight, accel (same as case 2)
    if maxSpeed(i) < maxSpeed(i+1)
        disp('case 2 or 4');
        % constant velocity data
        s = output(end, 2);
        distance = 0;
        if s < maxSpeed(i)
            output = cat(1, output,straightAccelFromSpeedMax(m, wd, Cd, Af, e, gr, tr, Cfx, Cfy, Cr, power, tStep, s, maxSpeed(i), t0, d0));
            distance = output(end, 4)-d0;
            t0 = output(end, 1);
            d0 = output(end, 4);
        end
        s = maxSpeed(i);
        distanceLeft = secLength(i)-distance;
        output = cat(1, output, constantVelocityData(s, distanceLeft, tStep, t0, d0));
        t0 = output(end, 1);
        d0 = output(end, 4);
    end
end

% compile output data
timeData = output(:,1);
velocityData = output(:,2);
accelData = output(:,3);
distanceData = output(:,4);
torqueData = output(:,5);

% graph output data

fig = figure;
x = timeData;
left_color = [0.5 0.5 0];
right_color = [0 0.5 0.5];
set(fig,'defaultAxesColorOrder',[left_color; right_color]);
title('Velocity/Accel vs Time');
xlabel('time (s)');


y = smoothdata(velocityData);
yyaxis left
plot(x,y)
ylabel('longitudinal velocity (m/s)');

z = smoothdata(accelData);
yyaxis right
plot(x,z)
ylabel('longitudinal acceleration (m/s^2)');


fig1 = figure;
x = timeData;
left_color = [0.5 0 0.5];
set(fig1,'defaultAxesColorOrder',[left_color]);
title('Torque vs Time');
a = smoothdata(torqueData);
plot(x,a)
xlabel('time (s)');
ylabel('Torque (Nm)');

end

