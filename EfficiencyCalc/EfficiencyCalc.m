
OLResults = SN3_60;

%Script to calculate overal car power consumption given OptimumLap 
%results. Factors in motor innefficiency map that OL does not support.


%Parameters
%OLResults: Matrix from OptimumLap with columns, Time, RPM, Torque, and Brake Power
%EffGraph: Matrix made from motor curves with coumns, RPM, Torque, and Efficiency
%numMotors: Number of Motors
    
EffRPM = E228(:,1);         %Given ref RPM (in RPM, NOT rad/s)
EffTorque = E228(:,2);      %(Nm)
Eff = E228(:,3);            %(%)

OLTime = OLResults(:,1);        %time (s)
OLRPM = OLResults(:,2);         %angular velocity (rad/s)
OLTorque = OLResults(:,3);      %Torque (Nm)
OLChargingData = OLResults(:,4);    %Battery Power (Kw)

RegenEff = .9;                  %Drivetrain Efficiency for regen    
LapLength = 1.06997;            %Length of One Lap (km)
InvEff = 0.97;                  %Inverter Efficiency
MaxCharge = 8;                  %Max Battery Charging Power (kW)


%scatteredInterpolant uses bicubic interpolation and linear extrapolation
InterpolEff = scatteredInterpolant(EffRPM*0.10472, EffTorque, Eff, "natural", "linear");
EffTime = InterpolEff(OLRPM, OLTorque)/100;


%Interpolating/Extrapolating eff at each time using biharmonic method
%{
count = 1:length(OLResults);
EffTime = zeros(length(count),1);
for n = count
    EffTime(n) = griddata(RPM,Torque,Eff,OLResults(n,2),OLResults(n,3)/numMotors,"v4");
end
%}


%TimeStep
TimeStep = zeros(length(OLTime),1);
for n = 1:(length(OLTime)-1)
    TimeStep(n) = OLTime(n+1) - OLTime(n);
end

%Base Energy Calc (useful for debugging)
BaseEnergy = cumtrapz(OLTime, OLRPM.*OLTorque);

%Tot Energy Calc
OLEnergySpent = cumtrapz(OLTime, OLRPM.*OLTorque./EffTime/InvEff);

%Battery Efficiency
OLChargingData = min(OLChargingData,MaxCharge);
OLCharged = cumtrapz(OLTime, OLChargingData*1000.*EffTime*RegenEff*InvEff);
        %times 1000 for W to kW

BaseEnergyTot = EnduranceEnergy(BaseEnergy, LapLength);
EnergyTot = EnduranceEnergy(OLEnergySpent, LapLength);
ChargeTot = EnduranceEnergy(OLCharged, LapLength);

FinalEnergyTot = EnergyTot-ChargeTot;

disp("    Base Energy w/o inefficiencies (kWh): " + BaseEnergyTot);
disp("    Energy Requirement w/o charging (kWh): " + EnergyTot);
disp("    Charging tot (kWh): " + ChargeTot);
disp("    Final Energy Requirement  (kWh): " + FinalEnergyTot);
disp("    Average Motor Efficiency (%):" + mean(EffTime))

%Takes vector of energy at each time step, sums them, and retursn MJ and kWh
function [kWh, MJ] = EnduranceEnergy(EnergyVec, LapLength)
    EnergyLap = EnergyVec(end)/10^6;
    EnergyTrack = EnergyLap*(22/LapLength);
    MJ = EnergyTrack;
    kWh = MJ*0.2777777778;
end