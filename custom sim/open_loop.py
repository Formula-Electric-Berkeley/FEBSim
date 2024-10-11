'''
Run a regular lap sim. Run simulate_endurance to run your simulation.
'''
import numpy as np
import lap_utils
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import pandas as pd

import matplotlib.pyplot as plt

# import our accumulator and motor models
import accumulator
import powertrain_model
motor = powertrain_model.motor()


# v_max is the electronically imposed maximum velocity; set to 999 by default
def simulate_mainloop(veh, tr):
    # Maximum speed curve
    v_max = np.zeros(tr.n, dtype=np.float32)
    bps_v_max = np.zeros(tr.n, dtype=np.float32)
    tps_v_max = np.zeros(tr.n, dtype=np.float32)
    

    # Find the maximum steady-state velocity at each mesh node
    for i in range(tr.n):
        v_max[i], tps_v_max[i], bps_v_max[i] = lap_utils.vehicle_model_lat(veh, tr, i)

    # Finding apexes
    apex, _ = find_peaks(-v_max) 
    
    v_apex = v_max[apex]  # Flipping to get positive values
    

    # Setting up standing start for open track configuration
    # TODO currently unused; we just have closed tracks atm
    if tr.config == 'Open' and False:
        if apex[0] != 0:  # If our first velocity max is not already an apex
            apex = np.insert(apex, 0, 1)  # Inject index 0 as apex
            v_apex = np.insert(v_apex, 0, 0)  # Inject standing start
        else:  # Index 1 is already an apex
            v_apex[0] = 0  # Set standing start at index 1
    
    # Checking if no apexes found and adding one if needed
    if len(apex)==0:
        apex = np.argmin(v_max)
        v_apex = v_max[apex]
    
    # Reordering apexes for solver time optimization
    old_apexes = v_apex #for plotting later
    apex_table = np.column_stack((v_apex, apex))
    apex_table = apex_table[apex_table[:, 0].argsort()]
    
    v_apex = apex_table[:, 0]
    apex = apex_table[:, 1]
    
    apex = apex.astype(int)

    # Getting driver inputs at apexes
    tps_apex = tps_v_max[apex]
    bps_apex = bps_v_max[apex]

    # Memory preallocation
    N = len(apex)  # Number of apexes
    flag = np.zeros((tr.n, 2), dtype=bool)  # Flag for checking that speed has been correctly evaluated
    v = np.full((tr.n, N, 2), np.inf, dtype=np.float32)
    ax = np.zeros((tr.n, N, 2), dtype=np.float32)
    ay = np.zeros((tr.n, N, 2), dtype=np.float32)
    tps = np.zeros((tr.n, N, 2), dtype=np.float32)
    bps = np.zeros((tr.n, N, 2), dtype=np.float32)
    
    counter = 0
    skipped = 0

    # Running simulation
    for k in range(2):  # Mode number; accel and decel
        mode = 1 if k == 0 else -1
        k_rest = 1 if k == 0 else 0
    
        for i in range(N):  # Apex number
            
            # Accelerate and decelerate from apex i to every other apex (j), unless... (see *) 

            if not (tr.config == 'Open' and mode == -1 and i == 0):  # Does not run in decel mode at standing start in open track
                
                # Getting other apex for later checking
                i_rest = lap_utils.other_points(i, N)
                
                # Getting apex index
                j = int(apex[i])

                # Saving speed, latacc, and driver inputs from presolved apex
                v[j, i, k] = v_apex[i]
                ay[j, i, k] = v_apex[i] ** 2 * tr.r[j]
                tps[j, :, 0] = np.full(N, tps_apex[i])
                tps[j, :, 1] = np.full(N, tps_apex[i])
                bps[j, :, 0] = np.full(N, bps_apex[i])
                bps[j, :, 1] = np.full(N, bps_apex[i])

                # Setting apex flag to show we've solved for this j node in the k direction
                flag[j, k] = True

                # Getting next point index (identify which j node will we solve for next)
                _, j_next = lap_utils.next_point(j, tr.n-1, mode)
    

                if not (tr.config == 'Open' and mode == 1 and i == 0):  # If we're not in standing start
                    # Assuming the same speed right after apex; initial guess
                    v[j_next, i, k] = v[j, i, k]

                    # Moving to the next point index
                    j_next, j = lap_utils.next_point(j, tr.n-1, mode)
    
                # Solve for the velocities at the rest of the j apexes 
                while True:
                    # Calculating speed, accelerations, and driver inputs from the vehicle model
                    v[j_next, i, k], ax[j, i, k], ay[j, i, k], tps[j, i, k], bps[j, i, k], overshoot = lap_utils.vehicle_model_comb(veh, tr, v[j, i, k], v_max[j_next], j, mode)

                    # Checking for limit
                    if overshoot:
                        #print("Overshot at ", j_next)
                        break

                    # Checking if the point is already solved in the other apex iteration
                    if flag[j, k] or flag[j, k_rest]:
                        # (*) if we've already solved for this j apex when starting from a different i, and if we found that velocity to be lower, 
                        # we know the rest of the velocities using our current i will not give a minimum velocity segment, so we go to a different i
                        if np.max(v[j_next, i, k] >= v[j_next, i_rest, k]) or np.max(v[j_next, i, k] > v[j_next, i_rest, k_rest]):
                            skipped += 1
                            break
                    
                    # Updating flag and Moving to the next point index
                    flag[j, k] = True 
                    j_next, j = lap_utils.next_point(j, tr.n-1, mode )
                    

                    # Checking if the lap is completed
                    if tr.config == 'Closed':
                        if j == apex[i]:  # Made it to the same apex
                            counter += 1
                            break
                    elif tr.config == 'Open':
                        if j == tr.n:  # Made it to the end
                            flag[j, k] = True #flag_update(flag, j, k, prg_size, logid, prg_pos)
                            break
                        if j == 1:  # Made it to the start
                            break


    # preallocation for results
    V = np.zeros(tr.n)
    AX = np.zeros(tr.n)
    AY = np.zeros(tr.n)
    TPS = np.zeros(tr.n)
    BPS = np.zeros(tr.n)
    
    # solution selection
    for i in range(tr.n):
        IDX = v.shape[1]
        min_values = np.min(v[i, :, :], axis=1)
        idx = np.argmin(min_values)
        V[i] = min_values[idx]

        if idx < IDX:  # Solved in acceleration
            AX[i] = ax[i, idx, 0]
            AY[i] = ay[i, idx, 0]
            TPS[i] = tps[i, idx, 0]
            BPS[i] = bps[i, idx, 0]
        else:  # Solved in deceleration
            AX[i] = ax[i, idx-IDX, 1]
            AY[i] = ay[i, idx-IDX, 1]
            TPS[i] = tps[i, idx-IDX, 1]
            BPS[i] = bps[i, idx-IDX, 1]

    return V, AX, AY, TPS, BPS

# The old simulate is now simulate_mainloop; this method holds the additional clutter and modifications; 9/22
def simulate(pack, tr):
    #Import track and vehicle files like OpenLap
    import vehicle as veh

    # Run our laputils functions to get first-pass solutions to the track
    V, AX, AY, TPS, BPS = simulate_mainloop(veh, tr)

    
    # laptime calculation
    dt = np.divide(tr.dx, V)
    time = np.cumsum(dt)
    laptime = time[-1]
    
    # TODO this seems dubious
    # Remove the final infinite value from V
    if (V[-1] == np.inf):
        V[-1] = V[-2]


    # TODO: 7/29/24; problem with 0 TPS, infinite V on endurance track showed up -- even in old versions, on new computer; this is a hotfix
    finite_velocities = V[np.isfinite(V)]
    minV = np.min(finite_velocities)
    V[np.isinf(V)] = minV
    

    '''Determination of forces for energy calculations'''

    # Interpolate wheel torque
    torque_func = interp1d(veh.vehicle_speed, veh.wheel_torque, kind='linear', fill_value='extrapolate')
    wheel_torque = TPS * torque_func(V)
    motor_torque = wheel_torque / (veh.ratio_primary*veh.ratio_gearbox*veh.ratio_final*veh.n_primary*veh.n_gearbox*veh.n_final)

    #Power regen
    brake_force = BPS/veh.beta                          # F_brake = BPS/veh.beta
    overall_regen_efficiency = 0.9                      # assumes xx% of energy from braking is regenerated by the motor
    regen_power_limit = 5                               # maximum rate of regen in kW
    regen_power_limit = regen_power_limit * 1000        # in Watts
    
    
    #TODO: model this better with more EECS data later

    peaks, __ = find_peaks(V)
    troughs, __ = find_peaks(-1*V)
    brake_energies = [0.0]

    for i in range(len(peaks)):
        # Find the next trough after the current peak
        next_troughs = troughs[troughs > peaks[i]]
        if next_troughs.size > 0:
            # Calculate the regenerated energy associated with this brake event
            time_of_brake = time[next_troughs[0]] - time[peaks[i]]              # total time of the brake event
            energy = 0.5*veh.M*(V[peaks[i]]**2 - V[next_troughs[0]]**2)         # kinetic energy lost during the brake event in J

            regen_energy_limit = regen_power_limit * time_of_brake              # maximum energy we can regen during this brake event

            if energy > regen_energy_limit:
                energy = regen_energy_limit

            brake_energies.append(energy)

    
    regenerated_energy = np.sum(brake_energies)*overall_regen_efficiency #in Joules

    wheel_speed = V/veh.tyre_radius # wheel speed in radians/second
    motor_speed = wheel_speed*veh.ratio_primary*veh.ratio_gearbox*veh.ratio_final   # convert to motor speed
    motor_power = motor_torque * motor_speed #in W


    # Replace motor_power and wheel_torque curves with zeros where values are negative
    motor_power = np.where(motor_power < 0, 0, motor_power)
    wheel_torque = np.where(wheel_torque < 0, 0, wheel_torque)

    # Work from the motor is force from the motor * dx integrated over the track
    motor_work = np.trapz(wheel_torque / veh.tyre_radius, x = tr.x)


    # Power of the motor numerically integrated over time
    motor_energy = np.sum(np.multiply(motor_power, dt))

    energy_cost_total = motor_work/(3.6*10**6)                   # in kWh
    energy_gained_total = regenerated_energy/(3.6*10**6)         # in kWh

    # RIT Energy Scalar
    E_predicted = 9.736 #kWh
    E_actual = 4.903
    energy_scale = E_actual/E_predicted

    # we predict like 
    #skidpad_scale_factor = 0.0063582105931661095 / 0.01552799231063735

    #scaled_motor_power = skidpad_scale_factor * motor_power


    energy_drain = energy_cost_total
    


    '''Modify our results to conform to electrical restrictions'''

    # Quasi-transient accumulator calculations
    # leave the option to not simulate with the pack
    pack_voltages = []
    pack_discharges = []
    base_speeds = []
    kvs = []
    mot_powers = []
    current_draws = []

    
    for i in range(len(motor_power)-1):
        power = motor_power[i]

        # drain the accumulator by the energy consumed during each time-step, and get our target current
        # Crucial point: if the breaker pops, we do not "drain" anything; we wait until we adjust the power and try again
        target_current, breaker_popped = pack.try_drain(power, dt[i])       
        # Motor power is instantaneous, so we can extract a current 
        # TODO: motor power * inverter_efficiency (97% or experimental / from library) = DC power
        # DC power = I * V -> gets I for pack
        # To get current going through the motor, we need to do more inverter magic 

        cell_data = pack.get_cell_data()

        # if we ever exceed our software-maxed current, decrease our power to target_power
        P = 0.05 # how quickly should we decrement TPS? TPS is zero to 1; reduce by 5% each time

        # While we're not commanding the proper motor power...
        while breaker_popped:

            # Reduce the torque for this segment to decrease our commanded power ; note this impacts lap time as well
            if TPS[i] > 0.0: #sanity check to make sure we're actually driving the motor
                tps_max = max(TPS[i] - P, 0) # if we run into 0 here, we have a problem
                # A lower TPS means a slower next-pass velocity
                V[i+1], TPS[i], BPS[i] = lap_utils.adjust_torque(veh, tr, tps_max, i, V[i], V[i+1])
            else:
                break
                 
            # Calculate motor power corresponding to this new torque command
            wheel_torque = TPS[i] * torque_func(V[i])        
            wheel_speed = V[i]/veh.tyre_radius 

            # find the new commanded power and error
            new_power = wheel_torque*wheel_speed

            # attempt to actually drain the accumulator and refresh the breaker if need-be (this is our escape from the loop)
            target_current, breaker_popped = pack.try_drain(new_power, dt[i])
            motor_power[i] = new_power

        # Recalculate energy consumption based on our adjusted motor powers
        motor_energy = np.sum(np.multiply(motor_power, dt))
        energy_cost_total = motor_energy/(3.6*10**6)                   # in kWh
        energy_drain = energy_cost_total


        # Get cell data for transient output
        cell_data = pack.get_cell_data()

        accumulator_voltage = cell_data["voltage"]
        pack_voltages.append(accumulator_voltage)
        pack_discharges.append(cell_data["discharge"])
        basespeed, kv_est = motor.calculate_base_speed(accumulator_voltage, power)
        mot_powers.append(power)
        base_speeds.append(basespeed)
        current_draws.append(target_current)

        # Adjust our velocity and laptime according to base speed limitations   
        # Effectively, reduce our maximum speed to the base speed at every point
        # TODO; this means we have to re-drain according to the new dt, but let's ignore that for now
        
        V[i] = min(V[i], basespeed)

    

    # re-calculate laptime with base speed limitation    
    dt = np.divide(tr.dx, V)
    time = np.cumsum(dt)
    laptime = time[-1]


    '''
    To optimize this properly, we should drain the accumulator after each point, then optimize accordingly
    Right now, we just look back at the lap, drain the accumulator, and adjust the speed/laptime "transiently" after we solve everything
    '''
    
            



    # Output all the data in a readable csv

    header = ['Time (s)', 'Velocity (m/s)', 'Torque Commanded (Nm)', 'Brake Commanded (N)', 
              'Pack Voltage (V)', 'Discharge (Wh)', 'Base Speed (m/s)', 'Motor Power (W)', 'Current Draw (A)']
    transient_output = pd.DataFrame(data=zip(time, V, motor_torque, brake_force, pack_voltages, pack_discharges, base_speeds, mot_powers, current_draws), columns=header)
    



    # Returns laptime in seconds, energy_drain in kWh
    return laptime, energy_drain, transient_output

# Simulate n laps of the currently loaded track
def simulate_laps(pack, numLaps, tr):
    laptime0, energistics0, output_df = simulate(pack, tr)

    laptimes = [laptime0]
    energy_drains = [energistics0]              #total energy drained from the accumulator (estimate)
    pack_failure = [pack.is_depleted()]         #check if the pack is depleted
    
    for lap in range(numLaps-1):
        laptime, energy_drain, output_df_prime = simulate(pack)
        output_df_prime["Time (s)"] = output_df_prime["Time (s)"] + sum(laptimes)
        laptimes.append(laptime)
        energy_drains.append(energy_drain)

        quick_breaker = pack.is_depleted()
        pack_failure.append(quick_breaker)
        output_df = pd.concat([output_df, output_df_prime])

        if quick_breaker:
            print("Pack failure on lap {}".format(lap+2))
            break

        
    

    # compile the basic outputs for optimization
    # THE DISCREPANCY WE SAW BETWEEN ACCUMULATOR DRAINAGE AND OPENLAP DRAINAGE IS BC OF REGEN
    total_time = np.sum(laptimes)
    total_energy_drain = np.sum(energy_drains)
    pack_failed = pack_failure[-1]


    return total_time, total_energy_drain, pack_failed, output_df



def simulate_pack(pack_data):
    pack = accumulator.Pack()
    pack.pack(pack_data[0], pack_data[1], pack_data[2]) 
    laptime, energy_drain, transient_output = simulate(pack)

    file_name = f"openloop_out.csv"
    transient_output.to_csv(file_name, sep=',', encoding='utf-8', index=False)





# RIT Actual
# 4.903 kWh
# 73.467

# RIT Predicted
# 8.3002 kWh
# 73.57

# Scale factor: 4.903 / 8.3002
# 0.590709
