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
import motor_model
motor = motor_model.motor()



def simulate(pack):
    #Import track and vehicle files like OpenLap
    import track as tr
    import vehicle as veh

    # Maximum speed curve
    v_max = np.zeros(tr.n, dtype=np.float32)
    bps_v_max = np.zeros(tr.n, dtype=np.float32)
    tps_v_max = np.zeros(tr.n, dtype=np.float32)
    
    for i in range(tr.n):
        v_max[i], tps_v_max[i], bps_v_max[i] = lap_utils.vehicle_model_lat(veh, tr, i)
    
    # HUD
    #print('Maximum speed calculated at all points.')
    
    # Finding apexes
    apex, _ = find_peaks(-v_max)  # Assuming findpeaks is a function that finds peaks in an array
    
    v_apex = v_max[apex]  # Flipping to get positive values
    

    # Setting up standing start for open track configuration
    # TODO currently unused; we just have closed tracks atm
    if tr.info.config == 'Open':
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
            if not (tr.info.config == 'Open' and mode == -1 and i == 0):  # Does not run in decel mode at standing start in open track
                # Getting other apex for later checking
                i_rest = lap_utils.other_points(i, N)
                # Getting apex index
                j = int(apex[i])
                #print("Here j = {}".format(j))
                # Saving speed, latacc, and driver inputs from presolved apex
                v[j, i, k] = v_apex[i]
                
                
                ay[j, i, k] = v_apex[i] ** 2 * tr.r[j]
                tps[j, :, 0] = np.full(N, tps_apex[i])
                tps[j, :, 1] = np.full(N, tps_apex[i])
                

                bps[j, :, 0] = np.full(N, bps_apex[i])
                bps[j, :, 1] = np.full(N, bps_apex[i])
                # Setting apex flag
                flag[j, k] = True
                # Getting next point index
                _, j_next = lap_utils.next_point(j, tr.n-1, mode)
    
                if not (tr.info.config == 'Open' and mode == 1 and i == 0):  # If not in standing start
                    # Assuming the same speed right after apex
                    v[j_next, i, k] = v[j, i, k]
                    # Moving to the next point index
                    j_next, j = lap_utils.next_point(j, tr.n-1, mode)
    
                while True:
                    # Calculating speed, accelerations, and driver inputs from the vehicle model
                    v[j_next, i, k], ax[j, i, k], ay[j, i, k], tps[j, i, k], bps[j, i, k], overshoot = lap_utils.vehicle_model_comb(veh, tr, v[j, i, k], v_max[j_next], j, mode)
                    #print("j = {} | | v = {}".format(j, v[j_next, i, k]))
                    # Checking for limit
                    if overshoot:
                        #print("Overshot")
                        break
                    # Checking if the point is already solved in the other apex iteration
                    if flag[j, k] or flag[j, k_rest]:
                        if np.max(v[j_next, i, k] >= v[j_next, i_rest, k]) or np.max(v[j_next, i, k] > v[j_next, i_rest, k_rest]):
                            skipped += 1
                            break
                    # Updating flag and progress bar
                    flag[j, k] = True #flag_update(flag, j, k, prg_size, logid, prg_pos)
                    # Moving to the next point index
                    j_next, j = lap_utils.next_point(j, tr.n-1, mode )
                    # Checking if the lap is completed

                    if tr.info.config == 'Closed':
                        if j == apex[i]:  # Made it to the same apex
                            counter += 1
                            break
                    elif tr.info.config == 'Open':
                        if j == tr.n:  # Made it to the end
                            flag[j, k] = True #flag_update(flag, j, k, prg_size, logid, prg_pos)
                            break
                        if j == 1:  # Made it to the start
                            break



    #print(v[:, 20, 0])
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

    #print(flag)
    #Below is a useful check to see where the apexes are:
    '''
    x = []
    xapex = []
    
    for i in range(len(v_max)):
        x.append(i)
        
        if i in apex:
            xapex.append(i)

    ax = plt.axes()
    #ax.plot(x, v_max, label="Max")
    #ax.plot(x, tr.r*50, label="1/Radius (arb.)")
    ax.scatter(xapex, old_apexes)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlabel("Longitudinal Coordinate x (m)")
    ax.set_title("Velocity Trace")
    
    
    #print(V)

    ax.plot(x, V, color='r', label="Final")
    ax.legend()
    plt.show()
    '''

    
    # laptime calculation    
    dt = np.divide(tr.dx, V)
    time = np.cumsum(dt)
    
    #max_sector = int(np.max(tr.sector))
    #sector_time = np.zeros(max_sector)
    
    #for i in range(1, max_sector + 1):
        #sector_time[i - 1] = np.max(time[tr.sector == i]) - np.min(time[tr.sector == i])
    
    laptime = time[-1]
    
    #print("Laptime is {}".format(laptime))
    
    
    
    # Remove the final infinite value from V
    if (V[-1] == np.inf):
        V[-1] = V[-2]

    # Interpolate wheel torque
    torque_func = interp1d(veh.vehicle_speed, veh.wheel_torque, kind='linear', fill_value='extrapolate')
    #print(V)
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


    

    '''
    Skidpad stuff
    
    
    L_skidpad = 2*np.pi*15.25       # total arc length of two laps of 15.25 m diameter skidpad

    L = L_skidpad + 1       # add the length of the initial straight

    N_skid_points = L / tr.mesh_size

    skid_dts = dt[:N_skid_points]
    skid_time = np.cumsum(skid_dts)
    
    # The total laptime for the skidpad
    skid_laptime = skid_time[-1]

    '''


    # Replace motor_power and wheel_torque curves with zeros where values are negative
    motor_power = np.where(motor_power < 0, 0, motor_power)
    wheel_torque = np.where(wheel_torque < 0, 0, wheel_torque)


    '''
    #TODO skidpad stuff

    skid_torques = wheel_torque[:N_skid_points]
    skid_work = np.trapz(skid_torques / veh.tyre_radius, x = tr.x[:N_skid_points])

    skid_energy_cost = skid_work/(3.6*10**6)                   # in kWh

    skidpad = True
    if skidpad:
        return skid_laptime, skid_energy_cost

    '''


    # Work from the motor is force from the motor * dx integrated over the track
    motor_work = np.trapz(wheel_torque / veh.tyre_radius, x = tr.x)


    # Power of the motor numerically integrated over time in kWh; gives values about 1 kWh larger
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

    motor_energy = np.sum(np.multiply(motor_power, dt))


    energy_cost_total = motor_energy/(3.6*10**6)                   # in kWh


    # so that we include the scaling upstream in our accumulator sims
    motor_power = motor_power


    
    # we don't want it for open_all, but it would be nice to have the scaled-up energy calcs
    #print("Energy cost is {:.3f} kWh".format(energy_scale*energy_cost_total*numLaps)) 
    #print("Regenerated {:.3f} kWh".format(energy_scale*energy_gained_total*numLaps)) 
    #print()

    energy_drain = energy_cost_total
    
    # information open_sweep needs to run accumulator calculations
    # laptime is for 1 lap, energy data is for the whole 22 laps


    # Quasi-transient accumulator calculations
    
    # leave the option to not simulate with the pack
    pack_voltages = []
    pack_discharges = []
    base_speeds = []
    kvs = []
    mot_powers = []
    current_draws = []

    for i, power in enumerate(motor_power):

        # drain the accumulator by the energy consumed during each time-step, and get our target current
        target_current, breaker_popped = pack.new_drain(power, dt[i])

        cell_data = pack.get_cell_data()

        # cconvert our target current to a target motor power; command a little less so that we converge
        target_power = target_current * cell_data["voltage"]*0.99 
        

        # if we ever exceed our software-maxed current, decrease our power to target_power
        error = target_power - power        # if error is negative (this should always should be the case), we need to *decrease* our power and thus our speed

        P = 1*10**-4 # convert from error (likely 1000s of Watts) to the velocity correction (likely 1s of m/s)

        while breaker_popped:
            # Change the velocity at this point to decrease our commanded power 
            V[i] = V[i] + error*P 

            # Calculate wheel torque and speed corresponding to this new velocity
            wheel_torque = TPS[i] * torque_func(V[i])        
            wheel_speed = V[i]/veh.tyre_radius 

            # find the new error
            new_power = wheel_torque*wheel_speed
            error = target_power - new_power

            #if error > np.abs(target_power - power): # if we get farther off
                #print("{}    {}".format(new_power, power))
                # it eventually converges due to the sign inclusion I put above
            

            # attempt to actually drain the accumulator and refresh the breaker if need-be (this is our escape from the loop)
            target_current, breaker_popped = pack.new_drain(new_power, dt[i])

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
    




    return laptime, energy_drain, transient_output
    


# Simulate 22 laps of endurance
def simulate_endurance(pack, numLaps):
    laptime0, energistics0, output_df = simulate(pack)

    laptimes = [laptime0]
    energy_drains = [energistics0]              #total energy drained from the accumulator (estimate)
    pack_failure = [pack.is_depleted()]         #check if the pack is depleted
    
    for lap in range(numLaps-1):
        
        laptime, energy_drain, output_df_prime = simulate(pack)
        laptimes.append(laptime)
        energy_drains.append(energy_drain)

        quick_breaker = pack.is_depleted()
        pack_failure.append(quick_breaker)
        output_df = pd.concat([output_df, output_df_prime])

        if quick_breaker:
            print("Pack failure on lap {}".format(lap+2))
            break

    



    
    #basic_header = ['Laptimes (s)', 'Energy Drains (kWh)', 'Pack Failure']
    #basic_output = pd.DataFrame(data=zip(laptimes, energy_drains, pack_failure), columns=basic_header)

    #header = ['Time (s)', 'Velocity (m/s)', 'Torque Commanded (Nm)', 'Brake Commanded (N)', 
    #          'Pack Voltage (V)', 'Discharge (Wh)', 'Base Speed (m/s)', 'Kv', 'Motor Power (W)', 'Iq Current (A)']

    #writer = pd.ExcelWriter('open_loop_out1.xlsx', engine='xlsxwriter')

    #output_df.to_excel(writer, sheet_name='Main Output', index=False, header=header)
    #basic_output.to_excel(writer, sheet_name='Basic Output', index=False, header=basic_header)

    #writer.close()
    

    # compile the basic outputs for optimization
    # THE DISCREPANCY WE SAW BETWEEN ACCUMULATOR DRAINAGE AND OPENLAP DRAINAGE IS BC OF REGEN
    total_time = np.sum(laptimes)
    total_energy_drain = np.sum(energy_drains)
    pack_failed = pack_failure[-1]


    return total_time, total_energy_drain, pack_failed



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
