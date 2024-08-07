%% OpenALL
% Sweeps a given variable running lap simulations for the specified
% vehicle and track

%% Clearing memory

close all force
diary('off')
fclose('all') ;

%% Starting timer

tic

%% Inputs

% Number of endurance laps (usually 22km / track length rounded to nearest
% whole number
numLaps = 21;

endurancefile = 'OpenTRACK Tracks/OpenTRACK_Michigan 2021 End_Closed_Forward.mat' ;
autoXfile = 'OpenTRACK Tracks/OpenTRACK_Michigan 2022_Open_Forward.mat' ;
vehiclefile = 'OpenVEHICLE Vehicles/OpenVEHICLE_FEB_SN3_30kW_Open Wheel.mat' ;
ptsfile  = 'SN3_Points_Reference.xlsx';

% Do you wish to sweep values? If false, given vehicle values will be used.
sweepBool = false;

% Variable to sweep. Run OpenLAP and see veh struct for vars available. Only
% vars that are a single value currently work (no motor curves).
sweepVar = "factor_drive";

% Values to sweep var with
vals2Sweep = 0.5;

% Do you want to sweep a second var? Set true if yes
sweep2 = false;

% Second var to sweep
sweepVar2 = "";

% Values to sweep second var. Use linspace and fill out first two values 
% (need same length as vals2Sweep
vals2Sweep2 = -3.5;

%% Loading variables

tr_end = load(endurancefile);
tr_autoX = load(autoXfile);
veh = load(vehiclefile);
ptsRef = readmatrix(ptsfile);

%% Export frequency

freq = 50 ; % [Hz]

%% Displaying Car Name

display("Results for: " + veh.name)

%% Running Lap Simulations over variable

% Checking if sweeping desired
if sweepBool

    % Check whether two sweep one or two variables. 
    if length(vals2Sweep) == 1 && (length(vals2Sweep2) == 1 || ~sweep2)
        % Simulating if only single value of variable(s) is given
        veh.(sweepVar) = vals2Sweep;
        
        if(sweep2)
            veh.(sweepVar2) = vals2Sweep2;
        end
    
        [sim_end] = simulate(veh,tr_end);
        tr_end = sim_end.laptime.data;
        e_end = (sim_end.energy.data(end-1)-sim_end.regen.data(end-1));

        [sim_autoX] = simulate(veh,tr_autoX);
        tr_autoX = sim_autoX.laptime.data;

        accTime = OpenDRAG(veh);

        pts = ptsCalc(ptsRef, numLaps, tr_end, tr_autoX, accTime, e_end);
        
        e_end = e_end*numLaps;

        %% Reporting results if single value
        display("Endurance Laptime: " + tr_end + " seconds")
        display("Endurance Energy with Regen: " + e_end + " kWh")
        display("AutoX Laptime: " + tr_autoX + " seconds")
        display("Acceleration Time: " + accTime + " seconds" + newline)

        display("Endurance points: " + pts(1))
        display("AutoX points: " + pts(2))
        display("Acceleration points: " + pts(3))
        display("Efficiency points: " + pts(4))
        display("Total Dynamic Points: " + pts(5) + " out of 500 available points")
        display("Total Points: " + pts(6) + " out of 600 available points *NOTE EFFICIENCY SCORING NOT ACCURATE*")

        display("Total Comp Points: " + (pts(6) + 270) + " out of 1000 available *ASSUMING 2023 FORECASTED RESULTS*")
    else 
        % If multiple values given

        % Check whether to sweep one or two variables
        if ~sweep2
            % Sweeping one variable

            %Initializing  lap time, and endurance energy vectors
            tVec_end  = zeros(length(vals2Sweep),1);
            eVec_end  = zeros(length(vals2Sweep),1);
            tVec_autoX  = zeros(length(vals2Sweep),1);
            eVec_autoX  = zeros(length(vals2Sweep),1);
            tVec_acc = zeros(length(vals2Sweep),1);

            pts  = zeros(length(vals2Sweep),6);
            
            %Run sim for each sweep value and record time and energy
            for i = 1:length(vals2Sweep); 
                veh.(sweepVar) = vals2Sweep(i);

                [sim_end] = simulate(veh,tr_end);
                tVec_end(i) = sim_end.laptime.data;
                eVec_end(i) = sim_end.energy.data(end-1)-sim_end.regen.data(end-1);

                [sim_autoX] = simulate(veh,tr_autoX);
                tVec_autoX(i) = sim_autoX.laptime.data;
                eVec_autoX(i) = sim_autoX.energy.data(end-1)-sim_autoX.regen.data(end-1);

                tVec_acc(i) = OpenDRAG(veh);

                pts(i,:) = ptsCalc(ptsRef, numLaps, tVec_end(i), tVec_autoX(i), tVec_acc(i), eVec_end(i));
            end
            
            eVec_end = eVec_end*numLaps;

            %Plot time and erergy versus swept value
            figure(1);
            set(gcf, 'Position',  [100, 100, 850, 1000])
            subplot(3,2,1)
            plot(vals2Sweep, tVec_end, "O-");
            title("Endurance Lap Time Vs. Swept " + sweepVar)
            xlabel(sweepVar)
            ylabel("Lap Time (s)")
            grid on;
            
            subplot(3,2,2)
            plot(vals2Sweep, eVec_end, "O-");
            title("Endurance Energy w/ Regen Vs. Swept " + sweepVar)
            xlabel(sweepVar)
            ylabel("End Energy (kWh)")
            grid on;

            subplot(3,1,2)
            plot(vals2Sweep, tVec_autoX, "O-");
            title("AutoX Lap Time Vs. Swept " + sweepVar)
            xlabel(sweepVar)
            ylabel("Lap Time (s)")
            grid on;

            subplot(3,1,3)
            plot(vals2Sweep, tVec_acc, "O-");
            title("Acceleration Time Vs. Swept " + sweepVar)
            xlabel(sweepVar)
            ylabel("Acc Time (s)")
            grid on;
            
            sgtitle("Times and Energies for: " + veh.name)

            %Point graphs
            figure(2);
            set(gcf, 'Position',  [1000, 100, 850, 1000])
            subplot(4,2,1)
            plot(vals2Sweep, pts(:,1), "O-");
            title("Endurance Pts Vs. Swept " + sweepVar)
            xlabel(sweepVar)
            ylabel("End Pts")
            grid on;

            subplot(4,2,3)
            plot(vals2Sweep, pts(:,2), "O-");
            title("Autocross Pts Vs. Swept " + sweepVar)
            xlabel(sweepVar)
            ylabel("AutoX Pts")
            grid on;

            subplot(4,2,5)
            plot(vals2Sweep, pts(:,3), "O-");
            title("Acceleration Pts Vs. Swept " + sweepVar)
            xlabel(sweepVar)
            ylabel("Acc Pts")
            grid on;

            subplot(4,2,7)
            plot(vals2Sweep, pts(:,4), "O-");
            title("Efficiency (*NOT ACCURATE*) Pts Vs. Swept " + sweepVar)
            xlabel(sweepVar)
            ylabel("Eff Pts")
            grid on;

            subplot(4,2,[2;4])
            plot(vals2Sweep, pts(:,5), "O-");
            title("End + AutoX + Acc Pts Vs. Swept " + sweepVar)
            xlabel(sweepVar)
            ylabel("Dynamic Pts")
            grid on;

            subplot(4,2,[6;8])
            plot(vals2Sweep, pts(:,6), "O-");
            title("End + AutoX + Acc + Eff Pts Vs. Swept " + sweepVar)
            xlabel(sweepVar)
            ylabel("All Pts")
            grid on;

            sgtitle("Points Plots for: " + veh.name)
        else
            %Sweeping 2 variables
            
            %Initializing End lap time, and endurance energy matrix
            tVec_end  = zeros(length(vals2Sweep),length(vals2Sweep2));
            eVec_end  = zeros(length(vals2Sweep),length(vals2Sweep2));

            %Initializing AutoX lap time, and autoX energy matrix
            tVec_autoX  = zeros(length(vals2Sweep),length(vals2Sweep2));
            eVec_autoX  = zeros(length(vals2Sweep),length(vals2Sweep2));

            %Intializing acceleration time matrix
            tVec_acc = zeros(length(vals2Sweep),length(vals2Sweep2));

            %Initializing pts 3D matrix. 
            pts  = zeros(length(vals2Sweep), length(vals2Sweep2), 6);
            
            %Run sim for each sweep value and record time and energy
            for i = 1:length(vals2Sweep); 
                for j = 1:length(vals2Sweep2)
                    veh.(sweepVar) = vals2Sweep(i);
                    veh.(sweepVar2) = vals2Sweep2(j);

                    [sim_end] = simulate(veh,tr_end);
                    tVec_end(i,j) = sim_end.laptime.data;
                    eVec_end(i,j) = sim_end.energy.data(end-1)-sim_end.regen.data(end-1);

                    [sim_autoX] = simulate(veh,tr_autoX);
                    tVec_autoX(i,j) = sim_autoX.laptime.data;
                    eVec_autoX(i,j) = sim_autoX.energy.data(end-1)-sim_autoX.regen.data(end-1);

                    accTime = OpenDRAG(veh);
                    tVec_acc(i,j) = accTime;

                    pts(i,j,:) = ptsCalc(ptsRef, numLaps, tVec_end(i,j), tVec_autoX(i,j), tVec_acc(i,j), eVec_end(i,j));
                end
            end
           
            eVec_end = eVec_end*numLaps;

            %Plot time and erergy versus swept value
            figure(1);
            set(gcf, 'Position',  [100, 100, 850, 1000])
            subplot(3,2,1)
            surf(vals2Sweep, vals2Sweep2, tVec_end');
            title("Endurance Lap Time Vs. Swept " + sweepVar + " and " + sweepVar2)
            xlabel(sweepVar)
            ylabel(sweepVar2)
            zlabel("Lap Time (s)")
            colorbar
            
            subplot(3,2,2)
            surf(vals2Sweep, vals2Sweep2, eVec_end');
            title("Endurance Energy w/ Regen Vs. Swept " + sweepVar + " and " + sweepVar2)
            xlabel(sweepVar)
            ylabel(sweepVar2)
            zlabel("End Energy (kWh)")
            colorbar

            subplot(3,1,2)
            surf(vals2Sweep, vals2Sweep2, tVec_autoX');
            title("AutoX Lap Time Vs. Swept " + sweepVar + " and " + sweepVar2)
            xlabel(sweepVar)
            ylabel(sweepVar2)
            zlabel("Lap Time (s)")
            colorbar

            subplot(3,1,3)
            surf(vals2Sweep, vals2Sweep2, tVec_acc');
            title("Acceleration  Time Vs. Swept " + sweepVar + " and " + sweepVar2)
            xlabel(sweepVar)
            ylabel(sweepVar2)
            zlabel("Acc Time (s)")
            colorbar

            sgtitle("Times and Energies for: " + veh.name)

            %Plot points
            figure(2);
            set(gcf, 'Position',  [1000, 100, 850, 1000])
            subplot(4,2,1)
            surf(vals2Sweep, vals2Sweep2, pts(:,:,1)');
            title("Endurance Pts Vs. Swept " + sweepVar + " and " + sweepVar2)
            xlabel(sweepVar)
            ylabel(sweepVar2)
            zlabel("End Pts")
            grid on;
            colorbar

            subplot(4,2,3)
            surf(vals2Sweep, vals2Sweep2, pts(:,:,2)');
            title("AutoX Pts Vs. Swept " + sweepVar + " and " + sweepVar2)
            xlabel(sweepVar)
            ylabel(sweepVar2)
            zlabel("AutoX Pts")
            grid on;
            colorbar

            subplot(4,2,5)
            surf(vals2Sweep, vals2Sweep2, pts(:,:,3)');
            title("Acc Pts Vs. Swept " + sweepVar + " and " + sweepVar2)
            xlabel(sweepVar)
            ylabel(sweepVar2)
            zlabel("Acc Pts")
            grid on;
            colorbar

            subplot(4,2,7)
            surf(vals2Sweep, vals2Sweep2, pts(:,:,4)');
            title("Eff Pts Vs. Swept " + sweepVar + " and " + sweepVar2)
            xlabel(sweepVar)
            ylabel(sweepVar2)
            zlabel("Eff Pts")
            grid on;
            colorbar

            subplot(4,2,[2;4])
            surf(vals2Sweep, vals2Sweep2, pts(:,:,5)');
            title("End + AutoX + Acc Pts Vs. Swept " + sweepVar + " and " + sweepVar2)
            xlabel(sweepVar)
            ylabel(sweepVar2)
            zlabel("Dynamic Pts")
            grid on;
            colorbar

            subplot(4,2,[6;8])
            surf(vals2Sweep, vals2Sweep2, pts(:,:,6)');
            title("End + AutoX + Acc + Eff Pts Vs. Swept " + sweepVar + " and " + sweepVar2)
            xlabel(sweepVar)
            ylabel(sweepVar2)
            zlabel("All Pts")
            grid on;
            colorbar

            sgtitle("Points Plots for: " + veh.name)

        end
    end
else
    %Simulate car with given vehicle file if sweeping is not desired
    %(sweepBool = false)
    [sim_end] = simulate(veh,tr_end);
    tr_end = sim_end.laptime.data;
    e_end = sim_end.energy.data(end-1)-sim_end.regen.data(end-1);

    [sim_autoX] = simulate(veh,tr_autoX);
    tr_autoX = sim_autoX.laptime.data;

    accTime = OpenDRAG(veh);

    pts = ptsCalc(ptsRef, numLaps, tr_end, tr_autoX, accTime, e_end);
    
    e_end = e_end*numLaps

    %% Reporting results if single value
    display("Endurance Laptime: " + tr_end + " seconds")
    display("Endurance Energy with Regen: " + e_end + " kWh")
    display("AutoX Laptime: " + tr_end + " seconds")
    display("Acc Time: " + accTime + " seconds")

    display("Endurance points: " + pts(1))
    display("AutoX points: " + pts(2))
    display("Acceleration points: " + pts(3))
    display("Efficiency points: " + pts(4))
    display("Total Dynamic Points: " + pts(5) + " out of 500 available points")
    display("Total Points: " + pts(6) + " out of 600 available points *NOTE EFFICIENCY SCORING NOT ACCURATE*")
end



%% Functions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sim] = simulate(veh,tr)
    
    %% initialisation
    
    % solver timer
    timer_solver_start = tic;
    
    %% maximum speed curve (assuming pure lateral condition)
    
    v_max = single(zeros(tr.n,1)) ;
    bps_v_max = single(zeros(tr.n,1)) ;
    tps_v_max = single(zeros(tr.n,1)) ;
    for i=1:tr.n
        [v_max(i),tps_v_max(i),bps_v_max(i)] = vehicle_model_lat(veh,tr,i) ;
    end
    
    %% finding apexes
    
    [v_apex,apex] = findpeaks(-v_max) ; % findpeaks works for maxima, so need to flip values
    v_apex = -v_apex ; % flipping to get positive values
    % setting up standing start for open track configuration
    if strcmp(tr.info.config,'Open')
        if apex(1)~=1 % if index 1 is not already an apex
            apex = [1;apex] ; % inject index 1 as apex
            v_apex = [0;v_apex] ; % inject standing start
        else % index 1 is already an apex
            v_apex(1) = 0 ; % set standing start at index 1
        end
    end
    % checking if no apexes found and adding one if needed
    if isempty(apex)
        [v_apex,apex] = min(v_max) ;
    end
    % reordering apexes for solver time optimisation
    apex_table = sortrows([v_apex,apex],1) ;
    v_apex = apex_table(:,1) ;
    apex = apex_table(:,2) ;
    % getting driver inputs at apexes
    tps_apex = tps_v_max(apex) ;
    bps_apex = bps_v_max(apex) ;
    
    %% simulation
    
    % memory preallocation
    N = uint32((length(apex))) ; % number of apexes
    flag = false(tr.n,2) ; % flag for checking that speed has been correctly evaluated
    % 1st matrix dimension equal to number of points in track mesh
    % 2nd matrix dimension equal to number of apexes
    % 3rd matrix dimension equal to 2 if needed (1 copy for acceleration and 1 for deceleration)
    v = single(inf*ones(tr.n,N,2)) ;
    ax = single(zeros(tr.n,N,2)) ;
    ay = single(zeros(tr.n,N,2)) ;
    tps = single(zeros(tr.n,N,2)) ;
    bps = single(zeros(tr.n,N,2)) ;
    
    % running simulation
    for i=1:N % apex number
        for k=uint8(1:2) % mode number
            switch k
                case 1 % acceleration
                    mode = 1 ;
                    k_rest = 2 ;
                case 2 % deceleration
                    mode = -1 ;
                    k_rest = 1 ;
            end
            if ~(strcmp(tr.info.config,'Open') && mode==-1 && i==1) % does not run in decel mode at standing start in open track
                % getting other apex for later checking
                [i_rest] = other_points(i,N) ;
                if isempty(i_rest)
                    i_rest = i ;
                end
                % getting apex index
                j = uint32(apex(i)) ;
                % saving speed & latacc & driver inputs from presolved apex
                v(j,i,k) = v_apex(i) ;
                ay(j,i,k) = v_apex(i)^2*tr.r(j) ;
                tps(j,:,1) = tps_apex(i)*ones(1,N) ;
                bps(j,:,1) = bps_apex(i)*ones(1,N) ;
                tps(j,:,2) = tps_apex(i)*ones(1,N) ;
                bps(j,:,2) = bps_apex(i)*ones(1,N) ;
                % setting apex flag
                flag(j,k) = true ;
                % getting next point index
                [~,j_next] = next_point(j,tr.n,mode,tr.info.config) ;
                if ~(strcmp(tr.info.config,'Open') && mode==1 && i==1) % if not in standing start
                    % assuming same speed right after apex
                    v(j_next,i,k) = v(j,i,k) ;
                    % moving to next point index
                    [j_next,j] = next_point(j,tr.n,mode,tr.info.config) ;
                end
                while 1
                    % calculating speed, accelerations and driver inputs from vehicle model
                    [v(j_next,i,k),ax(j,i,k),ay(j,i,k),tps(j,i,k),bps(j,i,k),overshoot] = vehicle_model_comb(veh,tr,v(j,i,k),v_max(j_next),j,mode) ;
                    % checking for limit
                    if overshoot
                        break
                    end
                    % checking if point is already solved in other apex iteration
                    if flag(j,k) || flag(j,k_rest)
                        if max(v(j_next,i,k)>=v(j_next,i_rest,k)) || max(v(j_next,i,k)>v(j_next,i_rest,k_rest))
                            break
                        end
                    end
                    % moving to next point index
                    [j_next,j] = next_point(j,tr.n,mode,tr.info.config) ;
                    % checking if lap is completed
                    switch tr.info.config
                        case 'Closed'
                            if j==apex(i) % made it to the same apex
                                break
                            end
                        case 'Open'
                            if j==tr.n % made it to the end
                                break
                            end
                            if j==1 % made it to the start
                                break
                            end
                    end
                end
            end
        end
    end
    
    %% post-processing resutls
    
    % result preallocation
    V = zeros(tr.n,1) ;
    AX = zeros(tr.n,1) ;
    AY = zeros(tr.n,1) ;
    TPS = zeros(tr.n,1) ;
    BPS = zeros(tr.n,1) ;
    % solution selection
    for i=1:tr.n
        IDX = length(v(i,:,1)) ;
        [V(i),idx] = min([v(i,:,1),v(i,:,2)]) ; % order of k in v(i,:,k) inside min() must be the same as mode order to not miss correct values
        if idx<=IDX % solved in acceleration
            AX(i) = ax(i,idx,1) ;
            AY(i) = ay(i,idx,1) ;
            TPS(i) = tps(i,idx,1) ;
            BPS(i) = bps(i,idx,1) ;
        else % solved in deceleration
            AX(i) = ax(i,idx-IDX,2) ;
            AY(i) = ay(i,idx-IDX,2) ;
            TPS(i) = tps(i,idx-IDX,2) ;
            BPS(i) = bps(i,idx-IDX,2) ;
        end
    end
    
    % laptime calculation
    if strcmp(tr.info.config,'Open')
        time = cumsum([tr.dx(2)./V(2);tr.dx(2:end)./V(2:end)]) ;
    else
        time = cumsum(tr.dx./V) ;
    end
    sector_time = zeros(max(tr.sector),1) ;
    for i=1:max(tr.sector)
        sector_time(i) = max(time(tr.sector==i))-min(time(tr.sector==i)) ;
    end
    laptime = time(end) ;
    
    % calculating forces
    M = veh.M ;
    g = 9.81 ;
    A = sqrt(AX.^2+AY.^2) ;
    Fz_mass = -M*g*cosd(tr.bank).*cosd(tr.incl) ;
    Fz_aero = 1/2*veh.rho*veh.factor_Cl*veh.Cl*veh.A*V.^2 ;
    Fz_total = Fz_mass+Fz_aero ;
    Fx_aero = 1/2*veh.rho*veh.factor_Cd*veh.Cd*veh.A*V.^2 ;
    Fx_roll = veh.Cr*abs(Fz_total) ;
    
    % calculating yaw motion, vehicle slip angle and steering input
    yaw_rate = V.*tr.r ;
    delta = zeros(tr.n,1) ;
    beta = zeros(tr.n,1) ;
    for i=1:tr.n
        B = [M*V(i)^2*tr.r(i)+M*g*sind(tr.bank(i));0] ;
        sol = veh.C\B ;
        delta(i) = sol(1)+atand(veh.L*tr.r(i)) ;
        beta(i) = sol(2) ;
    end
    steer = delta*veh.rack ;
    
    % calculating engine metrics
    wheel_torque = TPS.*interp1(veh.vehicle_speed,veh.wheel_torque,V,'linear','extrap') ;
    Fx_eng = wheel_torque/veh.tyre_radius ;
    engine_torque = TPS.*interp1(veh.vehicle_speed,veh.engine_torque,V,'linear','extrap') ;
    engine_power = TPS.*interp1(veh.vehicle_speed,veh.engine_power,V,'linear','extrap') ;
    engine_speed = interp1(veh.vehicle_speed,veh.engine_speed,V,'linear','extrap') ;
    gear = interp1(veh.vehicle_speed,veh.gear,V,'nearest','extrap') ;
    fuel_cons = cumsum(wheel_torque/veh.tyre_radius.*tr.dx/veh.n_primary/veh.n_gearbox/veh.n_final/veh.n_thermal/veh.fuel_LHV) ;
    fuel_cons_total = fuel_cons(end) ;

    
    % calculating kpis
    percent_in_corners = sum(tr.r~=0)/tr.n*100 ;
    percent_in_accel = sum(TPS>0)/tr.n*100 ;
    percent_in_decel = sum(BPS>0)/tr.n*100 ;
    percent_in_coast = sum(and(BPS==0,TPS==0))/tr.n*100 ;
    percent_in_full_tps = sum(tps==1)/tr.n*100 ;
    percent_in_gear = zeros(veh.nog,1) ;
    for i=1:veh.nog
        percent_in_gear(i) = sum(gear==i)/tr.n*100 ;
    end
    energy_spent_fuel = fuel_cons*veh.fuel_LHV ;
    energy_spent_mech = energy_spent_fuel*veh.n_thermal ;
    gear_shifts = sum(abs(diff(gear))) ;
    [~,i] = max(abs(AY)) ;
    ay_max = AY(i) ;
    ax_max = max(AX) ;
    ax_min = min(AX) ;
    sector_v_max = zeros(max(tr.sector),1) ;
    sector_v_min = zeros(max(tr.sector),1) ;
    for i=1:max(tr.sector)
        sector_v_max(i) = max(V(tr.sector==i)) ;
        sector_v_min(i) = min(V(tr.sector==i)) ;
    end

    %% Total 22km Endurance Energy Calcuation
    
    brake_force = BPS*veh.phi;
    regen_power = min(brake_force.*V,45000);
    
    tr_length = tr.x(end)/1000;
    
    energy = cumtrapz(time, engine_power)*2.77778e-7;
    regen = cumtrapz(time, regen_power)*2.77778e-7;
    
    %% saving results in sim structure
    %sim.sim_name.data = simname ;
    sim.distance.data = tr.x ;
    sim.distance.unit = 'm' ;
    sim.time.data = time ;
    sim.time.unit = 's' ;
    sim.N.data = N ;
    sim.N.unit = [] ;
    sim.apex.data = apex ;
    sim.apex.unit = [] ;
    sim.speed_max.data = v_max ;
    sim.speed_max.unit = 'm/s' ;
    sim.flag.data = flag ;
    sim.flag.unit = [] ;
    sim.v.data = v ;
    sim.v.unit = 'm/s' ;
    sim.Ax.data = ax ;
    sim.Ax.unit = 'm/s/s' ;
    sim.Ay.data = ay ;
    sim.Ay.unit = 'm/s/s' ;
    sim.tps.data = tps ;
    sim.tps.unit = [] ;
    sim.bps.data = bps ;
    sim.bps.unit = [] ;
    sim.elevation.data = tr.Z ;
    sim.elevation.unit = 'm' ;
    sim.speed.data = V ;
    sim.speed.unit = 'm/s' ;
    sim.yaw_rate.data = yaw_rate ;
    sim.yaw_rate.unit = 'rad/s' ;
    sim.long_acc.data = AX ;
    sim.long_acc.unit = 'm/s/s' ;
    sim.lat_acc.data = AY ;
    sim.lat_acc.unit = 'm/s/s' ;
    sim.sum_acc.data = A ;
    sim.sum_acc.unit = 'm/s/s' ;
    sim.throttle.data = TPS ;
    sim.throttle.unit = 'ratio' ;
    sim.brake_pres.data = BPS ;
    sim.brake_pres.unit = 'Pa' ;
    sim.brake_force.data = BPS*veh.phi ;
    sim.brake_force.unit = 'N' ;
    sim.steering.data = steer ;
    sim.steering.unit = 'deg' ;
    sim.delta.data = delta ;
    sim.delta.unit = 'deg' ;
    sim.beta.data = beta ;
    sim.beta.unit = 'deg' ;
    sim.Fz_aero.data = Fz_aero ;
    sim.Fz_aero.unit = 'N' ;
    sim.Fx_aero.data = Fx_aero ;
    sim.Fx_aero.unit = 'N' ;
    sim.Fx_eng.data = Fx_eng ;
    sim.Fx_eng.unit = 'N' ;
    sim.Fx_roll.data = Fx_roll ;
    sim.Fx_roll.unit = 'N' ;
    sim.Fz_mass.data = Fz_mass ;
    sim.Fz_mass.unit = 'N' ;
    sim.Fz_total.data = Fz_total ;
    sim.Fz_total.unit = 'N' ;
    sim.wheel_torque.data = wheel_torque ;
    sim.wheel_torque.unit = 'N.m' ;
    sim.engine_torque.data = engine_torque ;
    sim.engine_torque.unit = 'N.m' ;
    sim.engine_power.data = engine_power ;
    sim.engine_power.unit = 'W' ;
    sim.engine_speed.data = engine_speed ;
    sim.engine_speed.unit = 'rpm' ;
    sim.gear.data = gear ;
    sim.gear.unit = [] ;
    sim.fuel_cons.data = fuel_cons ;
    sim.fuel_cons.unit = 'kg' ;
    sim.fuel_cons_total.data = fuel_cons_total ;
    sim.fuel_cons_total.unit = 'kg' ;
    sim.laptime.data = laptime ;
    sim.laptime.unit = 's' ;
    sim.sector_time.data = sector_time ;
    sim.sector_time.unit = 's' ;
    sim.percent_in_corners.data = percent_in_corners ;
    sim.percent_in_corners.unit = '%' ;
    sim.percent_in_accel.data = percent_in_accel ;
    sim.percent_in_accel.unit = '%' ;
    sim.percent_in_decel.data = percent_in_decel ;
    sim.percent_in_decel.unit = '%' ;
    sim.percent_in_coast.data = percent_in_coast ;
    sim.percent_in_coast.unit = '%' ;
    sim.percent_in_full_tps.data = percent_in_full_tps ;
    sim.percent_in_full_tps.unit = '%' ;
    sim.percent_in_gear.data = percent_in_gear ;
    sim.percent_in_gear.unit = '%' ;
    sim.v_min.data = min(V) ;
    sim.v_min.unit = 'm/s' ;
    sim.v_max.data = max(V) ;
    sim.v_max.unit = 'm/s' ;
    sim.v_ave.data = mean(V) ;
    sim.v_ave.unit = 'm/s' ;
    sim.energy_spent_fuel.data = energy_spent_fuel ;
    sim.energy_spent_fuel.unit = 'J' ;
    sim.energy_spent_mech.data = energy_spent_mech ;
    sim.energy_spent_mech.unit = 'J' ;
    sim.gear_shifts.data = gear_shifts ;
    sim.gear_shifts.unit = [] ;
    sim.lat_acc_max.data = ay_max ;
    sim.lat_acc_max.unit = 'm/s/s' ;
    sim.long_acc_max.data = ax_max ;
    sim.long_acc_max.unit = 'm/s/s' ;
    sim.long_acc_min.data = ax_min ;
    sim.long_acc_min.unit = 'm/s/s' ;
    sim.sector_v_max.data = sector_v_max ;
    sim.sector_v_max.unit = 'm/s' ;
    sim.sector_v_min.data = sector_v_min ;
    sim.sector_v_min.unit = 'm/s' ;
    sim.energy.data = energy ;
    sim.energy.unit = 'kWh' ;
    sim.regen.data = regen ;
    sim.regen.unit = 'kWh' ;


    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v,tps,bps] = vehicle_model_lat(veh,tr,p)
    
    %% initialisation
    % getting track data
    g = 9.81 ;
    r = tr.r(p) ;
    incl = tr.incl(p) ;
    bank = tr.bank(p) ;
    factor_grip = tr.factor_grip(p)*veh.factor_grip ;
    % getting vehicle data
    factor_drive = veh.factor_drive ;
    factor_aero = veh.factor_aero ;
    driven_wheels = veh.driven_wheels ;
    % Mass
    M = veh.M ;
    % normal load on all wheels
    Wz = M*g*cosd(bank)*cosd(incl) ;
    % induced weight from banking and inclination
    Wy = -M*g*sind(bank) ;
    Wx = M*g*sind(incl) ;
    
    %% speed solution
    if r==0 % straight (limited by engine speed limit or drag)
        % checking for engine speed limit
        v = veh.v_max ;
        tps = 1 ; % full throttle
        bps = 0 ; % 0 brake
    else % corner (may be limited by engine, drag or cornering ability)
        %% initial speed solution
        % downforce coefficient
        D = -1/2*veh.rho*veh.factor_Cl*veh.Cl*veh.A ;
        % longitudinal tyre coefficients
        dmy = factor_grip*veh.sens_y ;
        muy = factor_grip*veh.mu_y ;
        Ny = veh.mu_y_M*g ;
        % longitudinal tyre coefficients
        dmx = factor_grip*veh.sens_x ;
        mux = factor_grip*veh.mu_x ;
        Nx = veh.mu_x_M*g ;
        % 2nd degree polynomial coefficients ( a*x^2+b*x+c = 0 )
        a = -sign(r)*dmy/4*D^2 ;
        b = sign(r)*(muy*D+(dmy/4)*(Ny*4)*D-2*(dmy/4)*Wz*D)-M*r ;
        c = sign(r)*(muy*Wz+(dmy/4)*(Ny*4)*Wz-(dmy/4)*Wz^2)+Wy ;
        % calculating 2nd degree polynomial roots
        if a==0
            v = sqrt(-c/b) ;
        elseif b^2-4*a*c>=0
            if (-b+sqrt(b^2-4*a*c))/2/a>=0
                v = sqrt((-b+sqrt(b^2-4*a*c))/2/a) ;
            elseif (-b-sqrt(b^2-4*a*c))/2/a>=0
                v = sqrt((-b-sqrt(b^2-4*a*c))/2/a) ;
            else
                error(['No real roots at point index: ',num2str(p)])
            end
        else
            error(['Discriminant <0 at point index: ',num2str(p)])
        end
        % checking for engine speed limit
        v = min([v,veh.v_max]) ;
        %% adjusting speed for drag force compensation
        adjust_speed = true ;
        while adjust_speed
            % aero forces
            Aero_Df = 1/2*veh.rho*veh.factor_Cl*veh.Cl*veh.A*v^2 ;
            Aero_Dr = 1/2*veh.rho*veh.factor_Cd*veh.Cd*veh.A*v^2 ;
            % rolling resistance
            Roll_Dr = veh.Cr*(-Aero_Df+Wz) ;
            % normal load on driven wheels
            Wd = (factor_drive*Wz+(-factor_aero*Aero_Df))/driven_wheels ;
            % drag acceleration
            ax_drag = (Aero_Dr+Roll_Dr+Wx)/M ;
            % maximum lat acc available from tyres
            ay_max = sign(r)/M*(muy+dmy*(Ny-(Wz-Aero_Df)/4))*(Wz-Aero_Df) ;
            % needed lat acc make turn
            ay_needed = v^2*r+g*sind(bank) ; % circular motion and track banking
            % calculating driver inputs
            if ax_drag<=0 % need throttle to compensate for drag
                % max long acc available from tyres
                ax_tyre_max_acc = 1/M*(mux+dmx*(Nx-Wd))*Wd*driven_wheels ;
                % getting power limit from engine
                ax_power_limit = 1/M*(interp1(veh.vehicle_speed,veh.factor_power*veh.fx_engine,v)) ;
                % available combined lat acc at ax_net==0 => ax_tyre==-ax_drag
                ay = ay_max*sqrt(1-(ax_drag/ax_tyre_max_acc)^2) ; % friction ellipse
                % available combined long acc at ay_needed
                ax_acc = ax_tyre_max_acc*sqrt(1-(ay_needed/ay_max)^2) ; % friction ellipse
                % getting tps value
                scale = min([-ax_drag,ax_acc])/ax_power_limit ;
                tps = max([min([1,scale]),0]) ; % making sure its positive
                bps = 0 ; % setting brake pressure to 0
            else % need brake to compensate for drag
                % max long acc available from tyres
                ax_tyre_max_dec = -1/M*(mux+dmx*(Nx-(Wz-Aero_Df)/4))*(Wz-Aero_Df) ;
                % available combined lat acc at ax_net==0 => ax_tyre==-ax_drag
                ay = ay_max*sqrt(1-(ax_drag/ax_tyre_max_dec)^2) ; % friction ellipse
                % available combined long acc at ay_needed
                ax_dec = ax_tyre_max_dec*sqrt(1-(ay_needed/ay_max)^2) ; % friction ellipse
                % getting brake input
                fx_tyre = max([ax_drag,-ax_dec])*M ;
                bps = max([fx_tyre,0])*veh.beta ; % making sure its positive
                tps = 0 ; % setting throttle to 0
            end
            % checking if tyres can produce the available combined lat acc
            if ay/ay_needed<1 % not enough grip
                v = sqrt((ay-g*sind(bank))/r)-1E-3 ; % the (-1E-3 factor is there for convergence speed)
            else % enough grip
                adjust_speed = false ;
            end
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v_next,ax,ay,tps,bps,overshoot] = vehicle_model_comb(veh,tr,v,v_max_next,j,mode)
    
    %% initialisation
    
    % assuming no overshoot
    overshoot = false ;
    % getting track data
    dx = tr.dx(j) ;
    r = tr.r(j) ;
    incl = tr.incl(j) ;
    bank = tr.bank(j) ;
    factor_grip = tr.factor_grip(j)*veh.factor_grip ;
    g = 9.81 ;
    % getting vehicle data
    if mode==1
        factor_drive = veh.factor_drive ;
        factor_aero = veh.factor_aero ;
        driven_wheels = veh.driven_wheels ;
    else
        factor_drive = 1 ;
        factor_aero = 1 ;
        driven_wheels = 4 ;
    end
    
    %% external forces
    
    % Mass
    M = veh.M ;
    % normal load on all wheels
    Wz = M*g*cosd(bank)*cosd(incl) ;
    % induced weight from banking and inclination
    Wy = -M*g*sind(bank) ;
    Wx = M*g*sind(incl) ;
    % aero forces
    Aero_Df = 1/2*veh.rho*veh.factor_Cl*veh.Cl*veh.A*v^2 ;
    Aero_Dr = 1/2*veh.rho*veh.factor_Cd*veh.Cd*veh.A*v^2 ;
    % rolling resistance
    Roll_Dr = veh.Cr*(-Aero_Df+Wz) ;
    % normal load on driven wheels
    Wd = (factor_drive*Wz+(-factor_aero*Aero_Df))/driven_wheels ;
    
    %% overshoot acceleration
    
    % maximum allowed long acc to not overshoot at next point
    ax_max = mode*(v_max_next^2-v^2)/2/dx ;
    % drag acceleration
    ax_drag = (Aero_Dr+Roll_Dr+Wx)/M ;
    % ovesrhoot acceleration limit
	ax_needed = ax_max-ax_drag ;
    
    %% current lat acc
    
    ay = v^2*r+g*sind(bank) ;
    
    %% tyre forces
    
    % longitudinal tyre coefficients
    dmy = factor_grip*veh.sens_y ;
    muy = factor_grip*veh.mu_y ;
    Ny = veh.mu_y_M*g ;
    % longitudinal tyre coefficients
    dmx = factor_grip*veh.sens_x ;
    mux = factor_grip*veh.mu_x ;
    Nx = veh.mu_x_M*g ;
    % friction ellipse multiplier
    if sign(ay)~=0 % in corner or compensating for banking
        % max lat acc available from tyres
        ay_max = 1/M*(sign(ay)*(muy+dmy*(Ny-(Wz-Aero_Df)/4))*(Wz-Aero_Df)+Wy) ;
        % max combined long acc available from tyres
        if abs(ay/ay_max)>1 % checking if vehicle overshot (should not happen, but check exists to exclude complex numbers in solution from friction ellipse)
            ellipse_multi = 0 ;
        else
            ellipse_multi = sqrt(1-(ay/ay_max)^2) ; % friction ellipse
        end
    else % in straight or no compensation for banking needed
        ellipse_multi = 1 ;
    end
    
    %% calculating driver inputs
    
    if ax_needed>=0 % need tps
        % max pure long acc available from driven tyres
        ax_tyre_max = 1/M*(mux+dmx*(Nx-Wd))*Wd*driven_wheels ;
        % max combined long acc available from driven tyres
        ax_tyre = ax_tyre_max*ellipse_multi ;
        % getting power limit from engine
        ax_power_limit = 1/M*(interp1(veh.vehicle_speed,veh.factor_power*veh.fx_engine,v,'linear',0)) ;
        % getting tps value
        scale = min([ax_tyre,ax_needed]/ax_power_limit) ;
        tps = max([min([1,scale]),0]) ; % making sure its positive
        bps = 0 ; % setting brake pressure to 0
        % final long acc command
        ax_com = tps*ax_power_limit ;
    else % need braking
        % max pure long acc available from all tyres
        ax_tyre_max = -1/M*(mux+dmx*(Nx-(Wz-Aero_Df)/4))*(Wz-Aero_Df) ;
        % max comb long acc available from all tyres
        ax_tyre = ax_tyre_max*ellipse_multi ;
        % tyre braking force
        fx_tyre = min(-[ax_tyre,ax_needed])*M ;
        % getting brake input
        bps = max([fx_tyre,0])*veh.beta ; % making sure its positive
        tps = 0 ; % seting throttle to 0
        % final long acc command
        ax_com = -min(-[ax_tyre,ax_needed]) ;
    end
    
    %% final results
    
    % total vehicle long acc
    ax = ax_com+ax_drag ;
    % next speed value
    v_next = sqrt(v^2+2*mode*ax*tr.dx(j)) ;
    % correcting tps for full throttle when at v_max on straights
    if tps>0 && v/veh.v_max>=0.999
        tps = 1 ;
    end
    
    %% checking for overshoot
    
    if v_next/v_max_next>1
        % setting overshoot flag
        overshoot = true ;
        % resetting values for overshoot
        v_next = inf ;
        ax = 0 ;
        ay = 0 ;
        tps = -1 ;
        bps = -1 ;
        return
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [j_next,j] = next_point(j,j_max,mode,tr_config)
    switch mode
        case 1 % acceleration
            switch tr_config
                case 'Closed'
                    if j==j_max-1
                        j = j_max ;
                        j_next = 1 ;
                    elseif j==j_max
                        j = 1 ;
                        j_next = j+1 ;
                    else
                        j = j+1 ;
                        j_next = j+1 ;
                    end
                case 'Open'
                    j = j+1 ;
                    j_next = j+1 ;
            end
        case -1 % deceleration
            switch tr_config
                case 'Closed'
                    if j==2
                        j = 1 ;
                        j_next = j_max ;
                    elseif j==1
                        j = j_max ;
                        j_next = j-1 ;
                    else
                        j = j-1 ;
                        j_next = j-1 ;
                    end
                case 'Open'
                    j = j-1 ;
                    j_next = j-1 ;
            end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [i_rest] = other_points(i,i_max)
    i_rest = (1:i_max)' ;
    i_rest(i) = [] ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [flag] = flag_update(flag,j,k,prg_size,logid,prg_pos)
    % current flag state
    p = sum(flag,'all')/size(flag,1)/size(flag,2) ;
    n_old = floor(p*prg_size) ; % old number of lines
    % new flag state
    flag(j,k) = true ;
    p = sum(flag,'all')/size(flag,1)/size(flag,2) ;
    n = floor(p*prg_size) ; % new number of lines
    % checking if state has changed enough to update progress bar
    if n>n_old
        progress_bar(flag,prg_size,logid,prg_pos) ;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = progress_bar(flag,prg_size,logid,prg_pos)
    % current flag state
    p = sum(flag,'all')/size(flag,1)/size(flag,2) ; % progress percentage
    n = floor(p*prg_size) ; % new number of lines
    e = prg_size-n ; % number of spaces
    % updating progress bar in command window
    fprintf(repmat('\b',1,prg_size+1+8)) % backspace to start of bar
    fprintf(repmat('|',1,n)) % writing lines
    fprintf(repmat(' ',1,e)) % writing spaces
    fprintf(']') % closing bar
    fprintf('%4.0f',p*100) % writing percentage
    fprintf(' [%%]') % writing % symbol
    % updating progress bar in log file
    fseek(logid,prg_pos,'bof') ; % start of progress bar position in log file
    fprintf(logid,'%s','Running: [') ;
    fprintf(logid,'%s',repmat('|',1,n)) ;
    fprintf(logid,'%s',repmat(' ',1,e)) ;
    fprintf(logid,'%s','] ') ;
    fprintf(logid,'%3.0f',p*100) ;
    fprintf(logid,'%s\n',' [%]') ;
    fseek(logid,0,'eof') ; % continue at end of file
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = disp_logo(logid)
    lg = [...
        '_______                    _____________________ ';...
        '__  __ \______________________  /___    |__  __ \';...
        '_  / / /__  __ \  _ \_  __ \_  / __  /| |_  /_/ /';...
        '/ /_/ /__  /_/ /  __/  / / /  /___  ___ |  ____/ ';...
        '\____/ _  .___/\___//_/ /_//_____/_/  |_/_/      ';...
        '       /_/                                       '...
        ] ;
    disp(lg) % command window
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = export_report(veh,tr,sim,freq,logid)
    % frequency
    freq = round(freq) ;
    % channel names
    all_names = fieldnames(sim) ;
    % number of channels to export
    S = 0 ;
    % channel id vector
    I = (1:length(all_names))' ;
    % getting only vector channels (excluding matrices)
    for i=1:length(all_names)
        % getting size for each channel
        s = size(eval(['sim.',all_names{i},'.data'])) ;
        % checking if channel is a vector
        if length(s)==2 && s(1)==tr.n && s(2)==1 % is vector
            S = S+1 ;
        else % is not vector
            I(i) = 0 ;
        end
    end
    % keeping only vector channel ids
    I(I==0) = [] ;
    % getting channel names
    channel_names = all_names(I)' ;
    % memory preallocation
    % data matrix
    data = single(zeros(tr.n,S)) ;
    % units vector
    channel_units = cell(1,length(I)) ;
    % getting data and units
    for i=1:length(I)
        data(:,i) = eval(['sim.',all_names{I(i)},'.data']) ;
        channel_units(i) = eval(['{sim.',all_names{I(i)},'.unit}']) ;
    end
    % new time vector for specified frequency
    t = (0:1/freq:sim.laptime.data)' ;
    % getting time channel id vector
    j = strcmp(string(channel_names),"time") ;
    % time data memory preallocation
    time_data = single(zeros(length(t),length(I))) ;
    % getting 
    for i=1:length(I)
         % checking if channel corresponds to time
        if i==j % time channel
            time_data(:,i) = t ;
        else % all other channels
            % checking for integer channel
            if strcmp(string(channel_names(i)),"gear") % gear needs to be integer
                time_data(:,i) = interp1(data(:,j),data(:,i),t,'nearest','extrap') ;
            else % all other channels are linearly interpolated
                time_data(:,i) = interp1(data(:,j),data(:,i),t,'linear','extrap') ;
            end
        end
    end
    % opening and writing .csv file
    % HUD
    disp('Export initialised.')
    fprintf(logid,'%s\n','Export initialised.') ;
    % filename
    filename = sim.sim_name.data+".csv" ;
    % opening file
    fid = fopen(filename,'w') ;
    % writing file header
    fprintf(fid,'%s,%s\n',["Format","OpenLAP Export"]) ;
    fprintf(fid,'%s,%s\n',["Venue",tr.info.name]) ;
    fprintf(fid,'%s,%s\n',["Vehicle",veh.name]) ;
    fprintf(fid,'%s,%s\n',["Driver",'OpenLap']) ;
    fprintf(fid,'%s\n',"Device") ;
    fprintf(fid,'%s\n',"Comment") ;
    fprintf(fid,'%s,%s\n',["Date",datestr(now,'dd/mm/yyyy')]) ;
    fprintf(fid,'%s,%s\n',["Time",datestr(now,'HH:MM:SS')]) ;
    fprintf(fid,'%s,%s\n',["Frequency",num2str(freq,'%d')]) ;
    fprintf(fid,'\n') ;
    fprintf(fid,'\n') ;
    fprintf(fid,'\n') ;
    fprintf(fid,'\n') ;
    fprintf(fid,'\n') ;
    % writing channels
    form = [repmat('%s,',1,length(I)-1),'%s\n'] ;
    fprintf(fid,form,channel_names{:}) ;
    fprintf(fid,form,channel_names{:}) ;
    fprintf(fid,form,channel_units{:}) ;
    fprintf(fid,'\n') ;
    fprintf(fid,'\n') ;
    form = [repmat('%f,',1,length(I)-1),'%f\n'] ;
    for i=1:length(t)
        fprintf(fid,form,time_data(i,:)) ;
    end
    % closing file
    fclose(fid) ;
    % HUD
    disp('Exported .csv file successfully.')
    fprintf(logid,'%s\n','Exported .csv file successfully.') ;
end

function accTime = OpenDRAG(veh)
    %% Simulation settings
    
    % date and time in simulation name
    use_date_time_in_name = false ;
    % time step
    dt = 1E-3 ;
    % maximum simulation time for memory preallocation
    t_max = 60 ;
    % acceleration sensitivity for drag limitation
    ax_sens = 0.05 ; % [m/s2]
    % speed traps
    speed_trap = [50;100;150;200;250;300;350]/3.6 ;
    % track data
    bank = 0 ;
    incl = 0 ;
    
    %% Vehicle data preprocessing
    
    % mass
    M = veh.M ;
    % gravity constant
    g = 9.81 ;
    % longitudinal tyre coefficients
    dmx = veh.factor_grip*veh.sens_x ;
    mux = veh.factor_grip*veh.mu_x ;
    Nx = veh.mu_x_M*g ;
    % normal load on all wheels
    Wz = M*g*cosd(bank)*cosd(incl) ;
    % induced weight from banking and inclination
    Wy = M*g*sind(bank) ;
    Wx = M*g*sind(incl) ;
    % ratios
    rf = veh.ratio_final ;
    rg = veh.ratio_gearbox ;
    rp = veh.ratio_primary ;
    % tyre radius
    Rt = veh.tyre_radius ;
    % drivetrain efficiency
    np = veh.n_primary ;
    ng = veh.n_gearbox ;
    nf = veh.n_final ;
    % engine curves
    rpm_curve = [0;veh.en_speed_curve] ;
    torque_curve = veh.factor_power*[veh.en_torque_curve(1);veh.en_torque_curve] ;
    % shift points
    shift_points = table2array(veh.shifting(:,1)) ;
    shift_points = [shift_points;veh.en_speed_curve(end)] ;
    
    %% Acceleration preprocessing
    
    % memory preallocation
    N = t_max/dt ;
    T = -ones(N,1) ;
    X = -ones(N,1) ;
    V = -ones(N,1) ;
    A = -ones(N,1) ;
    RPM = -ones(N,1) ;
    TPS = -ones(N,1) ;
    BPS = -ones(N,1) ;
    GEAR = -ones(N,1) ;
    MODE = -ones(N,1) ;
    % initial time
    t = 0 ;
    t_start = 0 ;
    % initial distance
    x = 0 ;
    x_start = 0 ;
    % initial velocity
    v = 0 ;
    % initial accelerartion
    a = 0 ;
    % initial gears
    gear = 1 ;
    gear_prev = 1 ;
    % shifting condition
    shifting = false ;
    % initial rpm
    rpm = 0 ;
    % initial tps
    tps = 0 ;
    % initial bps
    bps = 0 ;
    % initial trap number
    trap_number = 1 ;
    % speed trap checking condition
    check_speed_traps = true ;
    % iteration number
    i = 1 ;
    
    %% Acceleration
    
    while true
        % saving values
        MODE(i) = 1 ;
        T(i) = t ;
        X(i) = x ;
        V(i) = v ;
        A(i) = a ;
        RPM(i) = rpm ;
        TPS(i) = tps ;
        BPS(i) = 0 ;
        GEAR(i) = gear ;
        % checking if rpm limiter is on or if out of memory
        if v>=veh.v_max
            break
        elseif i==N
            % HUD
            disp(['Did not reach maximum speed at time ',num2str(t),' s'])
            break
        end
        % check if drag limited
        if tps==1 && ax+ax_drag<=ax_sens
            break
        end
        % checking speed trap
        if check_speed_traps
            % checking if current speed is above trap speed
            if v>=speed_trap(trap_number)
                % next speed trap
                trap_number = trap_number+1 ;
                % checking if speed traps are completed
                if trap_number>length(speed_trap)
                    check_speed_traps = false ;
                end
            end
        end
        % aero forces
        Aero_Df = 1/2*veh.rho*veh.factor_Cl*veh.Cl*veh.A*v^2 ;
        Aero_Dr = 1/2*veh.rho*veh.factor_Cd*veh.Cd*veh.A*v^2 ;
        % rolling resistance
        Roll_Dr = veh.Cr*(-Aero_Df+Wz) ;
        % normal load on driven wheels
        Wd = (veh.factor_drive*Wz+(-veh.factor_aero*Aero_Df))/veh.driven_wheels ;
        % drag acceleration
        ax_drag = (Aero_Dr+Roll_Dr+Wx)/M ;
        % rpm calculation
        if gear==0 % shifting gears
            rpm = rf*rg(gear_prev)*rp*v/Rt*60/2/pi ;
            rpm_shift = shift_points(gear_prev) ;
        else % gear change finished
            rpm = rf*rg(gear)*rp*v/Rt*60/2/pi ;
            rpm_shift = shift_points(gear) ;
        end
        % checking for gearshifts
        if rpm>=rpm_shift && ~shifting % need to change gears
            if gear==veh.nog % maximum gear number
                break
            else % higher gear available
                % shifting condition
                shifting = true ;
                % shift initialisation time
                t_shift = t ;
                % zeroing  engine acceleration
                ax = 0 ;
                % saving previous gear
                gear_prev = gear ;
                % setting gear to neutral for duration of gearshift
                gear = 0 ;
            end
        elseif shifting % currently shifting gears
            % zeroing  engine acceleration
            ax = 0 ;
            % checking if gearshift duration has passed
            if t-t_shift>veh.shift_time
                % shifting condition
                shifting = false ;
                % next gear
                gear = gear_prev+1 ;
            end
        else % no gearshift
            % max long acc available from tyres
            ax_tyre_max_acc = 1/M*(mux+dmx*(Nx-Wd))*Wd*veh.driven_wheels ;
            % getting power limit from engine
            engine_torque = interp1(rpm_curve,torque_curve,rpm) ;
            wheel_torque = engine_torque*rf*rg(gear)*rp*nf*ng*np ;
            ax_power_limit = 1/M*wheel_torque/Rt ;
            % final long acc
            ax = min([ax_power_limit,ax_tyre_max_acc]) ;
        end
        % tps
        tps = ax/ax_power_limit ;
        % longitudinal acceleration
        a = ax+ax_drag ;
        % new position
        x = x+v*dt+1/2*a*dt^2 ;
        % new velocity
        v = v+a*dt ;
        % new time
        t = t+dt ;
        % next iteration
        i = i+1 ;
    end
    i_acc = i ; % saving acceleration index
    % average acceleration
    a_acc_ave = v/t ;
    %disp(['Average acceleration:    ',num2str(a_acc_ave/9.81,'%6.3f'),' [G]'])
    %disp(['Peak acceleration   :    ',num2str(max(A)/9.81,'%6.3f'),' [G]'])
    
    %% Deceleration preprocessing
    
    % saving time and position of braking start
    t_start = t ;
    x_start = x ;
    % speed trap condition
    check_speed_traps = true ;
    % active braking speed traps
    speed_trap_decel = speed_trap(speed_trap<=v) ;
    trap_number = length(speed_trap_decel) ;
    
    %% Deceleration
    
    while true
        % saving values
        MODE(i) = 2 ;
        T(i) = t ;
        X(i) = x ;
        V(i) = v ;
        A(i) = a ;
        RPM(i) = rpm ;
        TPS(i) = 0 ;
        BPS(i) = bps ;
        GEAR(i) = gear ;
        % checking if stopped or if out of memory
        if v<=0
            % zeroing speed
            v = 0 ;
            break
        elseif i==N
            % HUD
            disp(['Did not stop at time ',num2str(t),' s'])
            break
        end
        % checking speed trap
        if check_speed_traps
            % checking if current speed is under trap speed
            if v<=speed_trap_decel(trap_number)
                % next speed trap
                trap_number = trap_number-1 ;
                % checking if speed traps are completed
                if trap_number<1
                    check_speed_traps = false ;
                end
            end
        end
        % aero forces
        Aero_Df = 1/2*veh.rho*veh.factor_Cl*veh.Cl*veh.A*v^2 ;
        Aero_Dr = 1/2*veh.rho*veh.factor_Cd*veh.Cd*veh.A*v^2 ;
        % rolling resistance
        Roll_Dr = veh.Cr*(-Aero_Df+Wz) ;
        % drag acceleration
        ax_drag = (Aero_Dr+Roll_Dr+Wx)/M ;
        % gear
        gear = interp1(veh.vehicle_speed,veh.gear,v) ;
        % rpm
        rpm = interp1(veh.vehicle_speed,veh.engine_speed,v) ;
        % max long dec available from tyres
        ax_tyre_max_dec = -1/M*(mux+dmx*(Nx-(Wz-Aero_Df)/4))*(Wz-Aero_Df) ;
        % final long acc
        ax = ax_tyre_max_dec ;
        % brake pressure
        bps = -veh.beta*veh.M*ax ;
        % longitudinal acceleration
        a = ax+ax_drag ;
        % new position
        x = x+v*dt+1/2*a*dt^2 ;
        % new velocity
        v = v+a*dt ;
        % new time
        t = t+dt ;
        % next iteration
        i = i+1 ;
    end
    % average deceleration
    a_dec_ave = V(i_acc)/(t-t_start) ;
    %disp(['Average deceleration:    ',num2str(a_dec_ave/9.81,'%6.3f'),' [G]'])
    %disp(['Peak deceleration   :    ',num2str(-min(A)/9.81,'%6.3f'),' [G]'])

    
    %% Results compression
    
    % getting values to delete
    to_delete = T==-1 ;
    % deleting values
    T(to_delete) = [] ;
    X(to_delete) = [] ;
    V(to_delete) = [] ;
    A(to_delete) = [] ;
    RPM(to_delete) = [] ;
    TPS(to_delete) = [] ;
    BPS(to_delete) = [] ;
    GEAR(to_delete) = [] ;
    MODE(to_delete) = [] ;
    

    
    %% 75m time
    
    %Find index and time at which front of car hits  75m
    acc_75m_t = T(find(X > 75,1));
    
    %Timer starts when car passes line so find time when back of car passes line
    carlength = 2.250; %(m)
    acc_headstart_t = T(find(X > carlength,1));
    
    accTime = acc_75m_t - acc_headstart_t;
end

%% Function to calculate points of endurance, autocross, acceleration and efficiency events
function pts = ptsCalc(ptsRef, numLaps, tEnd, tAutoX, tAcc, eEnd)

    % Reading reference values from ptsRef file
    tMinEnd = ptsRef(1,2);
    tMinAutoX = ptsRef(2,2);
    tMinAcc = ptsRef(3,2);
    eMinEnd = ptsRef(4,2);
    eFactorMin = ptsRef(5,2);
    eFactorMax = ptsRef(6,2);

    % Adjust endurance time and energy for number of laps
    tEnd = tEnd*numLaps;
    eEnd = eEnd*numLaps;

    % Calculate points for Endurance, Autocross, and Acceleration (from
    % rulebook)
    ptsEnd = min(250, 250*((tMinEnd*1.45/tEnd)-1)/((tMinEnd*1.45/tMinEnd)-1));
    ptsAutoX = min(125, 118.5*((tMinAutoX*1.45/tAutoX)-1)/((tMinAutoX*1.45/tMinAutoX)-1) + 6.5);
    ptsAcc = min(100, 95.5*((tMinAcc*1.5/tAcc)-1)/((tMinAcc*1.5/tMinAcc)-1) + 4.5);

    % Calculate efficiency factor and points
    effFactor = (tMinEnd/tEnd)*(eMinEnd/eEnd);
    ptsEff = min(100, 100*(effFactor*eFactorMin)/(eFactorMax*eFactorMin));


    %Return calculated points
    pts = [ptsEnd; ptsAutoX; ptsAcc; ptsEff; 0; 0];

    %Sum dynamic points
    pts(5) = sum(pts(1:3));

    %Sum all points
    pts(6) = sum(pts(1:4));
end