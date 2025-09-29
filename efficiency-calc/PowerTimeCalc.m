    function [PowerTime,OLPower] = EfficiencyCalc(OLResults, EffGraph,numMotors,TimeStep)
    %Parameters
        %OLResults: Matrix from OptimumLap with columns, Time, RPM, Torque, and Brake Power
        %EffGraph: Matrix made from motor curves with columns, RPM, Torque, and Efficiency
        %numMotors: Number of Motors
        
    EffRPM = EffGraph(:,1);
    EffTorque = EffGraph(:,2);
    Eff = EffGraph(:,3);
    
    OLTime = OLResults(:,1);        %time (s)
    OLRPM = OLResults(:,2);         %angular velocity (rad/s)
    OLTorque = OLResults(:,3);      %Torque (Nm)
    OLCharging = OLResults(:,4);    %Battery Power (Kw)
    
    RegenEff = 1;                   %Drivetrain Efficiency for regen   
    TrackLength = 1.069;            %Length of One Lap (km)
    InvEff = 0.97;                  %Inverter Efficiency


    %scatteredInterpolant uses bicubic interpolation and linear extrapolation
    InterpolEff = scatteredInterpolant(EffRPM/20,EffTorque,Eff,"natural","linear");
        %NOTE: dividing RPM by 20 improves interpolation results
    EffTime = InterpolEff(OLRPM*9.5493/20,OLTorque);
    
    
    %Interpolating/Extrapolating eff at each time using biharmonic method
    %{
    count = 1:length(OLResults);
    EffTime = zeros(length(count),1);
    for n = count
        EffTime(n) = griddata(RPM,Torque,Eff,OLResults(n,2),OLResults(n,3)/numMotors,"v4");
    end
    %}
    
    
    %OLTimeStep
    OLTimeStep = zeros(length(OLTime),1);
    for n = 1:(length(OLTime)-1)
        OLTimeStep(n) = OLTime(n+1) - OLTime(n);
    end
    OLTimeStep(end) = OLTimeStep(end-1);
    
    %Battery Efficiency
    OLCharging = OLCharging.*EffTime/100*RegenEff*InvEff/2;
    
    %{
    %Tot Energy Calc
    Energylap = sum(OLRPM.*OLTorque./(PowerTime/100).*TimeStep/InvEff-OLCharging)/(10^6);
    Energytot = Energylap*(22/TrackLength);
    %}
    
    
    OLPower = (OLRPM.*OLTorque./(EffTime/100)/InvEff-OLCharging)/1000;
    OLPower = OLPower(1:length(OLPower));
    
    %New Time Step
    NTimeSteps = ceil(max(OLTime)/TimeStep);
    PowerTime = zeros(NTimeSteps,2);
    PowerTime(:,2) = 0:TimeStep:TimeStep*(NTimeSteps-1);
    
    %Interpolating Power at Each Step
    PowerTime(:,1) = interp1(OLTime,OLPower,PowerTime(:,2),'spline');
    
end