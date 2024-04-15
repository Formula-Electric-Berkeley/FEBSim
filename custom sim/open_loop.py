'''
This will be the main file
'''
import numpy as np
import lap_utils
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt



#Track and vehicle files are located in the respective python scripts


def simulate():
    #Import track and vehicle files like OpenLap (every time we call simulate, reload track and vehicle)
    import track_files.track as tr
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
    if tr.info.config == 'Open':
        if apex[0] != 1:  # If index 1 is not already an apex
            apex = np.insert(apex, 0, 1)  # Inject index 1 as apex
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
    '''
    OpenAll

    Sweep multiple variables;
    vary the mass and power cap
    output the energy consumption and laptime
    
    '''
    

    # Getting driver inputs at apexes
    tps_apex = tps_v_max[apex]
    bps_apex = bps_v_max[apex]
    
    
    
    # Memory preallocation
    N = len(apex)  # Number of apexes
    flag = np.zeros((tr.n, 2), dtype=bool)  # Flag for checking that speed has been correctly evaluated
    v = np.full((tr.n, N, 2), np.inf, dtype=np.float32)
    ax = np.zeros((tr.n, N, 2), dtype=np.float32)
    ay = np.zeros((tr.n, N, 2), dtype=np.float32)
    tps = np.zeros((tr.n, N), dtype=np.float32)
    bps = np.zeros((tr.n, N), dtype=np.float32)
    
    counter = 0
    skipped = 0
    
    # Running simulation
    for k in range(2):  # Mode number
        mode = 1 if k == 0 else -1
        k_rest = 1 if k == 0 else 0
    
        for i in range(N):  # Apex number
            if not (tr.info.config == 'Open' and mode == -1 and i == 0):  # Does not run in decel mode at standing start in open track
                # Getting other apex for later checking
                i_rest = lap_utils.other_points(i, N)
                # Getting apex index
                j = int(apex[i])

                # Saving speed, latacc, and driver inputs from presolved apex
                v[j, i, k] = v_apex[i]
                
                
                ay[j, i, k] = v_apex[i] ** 2 * tr.r[j]
                tps[j, :] = tps_apex[i] * np.ones(N)
                bps[j, :] = bps_apex[i] * np.ones(N)
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
                    v[j_next, i, k], ax[j, i, k], ay[j, i, k], tps[j, k], bps[j, k], overshoot = lap_utils.vehicle_model_comb(veh, tr, v[j, i, k], v_max[j_next], j, mode)

                    # Checking for limit
                    if overshoot:
                        #print("Overshot")
                        break
                    # Checking if the point is already solved in the other apex iteration
                    if flag[j, k]:
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

    #print("Counter: {}".format(counter))
    #print("Skipped: {}".format(skipped))

    # preallocation for results
    V = np.zeros(tr.n)
    AX = np.zeros(tr.n)
    AY = np.zeros(tr.n)
    TPS = np.zeros(tr.n)
    BPS = np.zeros(tr.n)
    
    # solution selection
    for i in range(tr.n):
        IDX = v.shape[1]
        #min_values = np.min([v[i, :, 0], v[i, :, 1]], axis=0)
        #idx = np.argmin(min_values)
        #V[i] = min_values[idx]
        min_values = np.min(v[i, :, :], axis=1)
        idx = np.argmin(min_values)
    
        V[i] = min_values[idx]
    
        if idx <= IDX:  # solved in acceleration
            AX[i] = ax[i, idx, 0]
            AY[i] = ay[i, idx, 0]
            TPS[i] = tps[i, 0]
            BPS[i] = bps[i, 0]
        else:  # solved in deceleration
            AX[i] = ax[i, idx - IDX, 1]
            AY[i] = ay[i, idx - IDX, 1]
            TPS[i] = tps[i, 1]
            BPS[i] = bps[i, 1]
        
            
    
    #Below is a useful check to see where the apexes are:
    '''
    x = []
    xapex = []
    
    for i in range(len(v_max)):
        x.append(i)
        
        if i in apex:
            xapex.append(i)

    ax = plt.axes()
    ax.plot(x, v_max, label="Max")
    ax.plot(x, tr.r*50, label="1/Radius (arb.)")
    ax.scatter(xapex, old_apexes)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlabel("Longitudinal Coordinate x (m)")
    ax.set_title("Velocity Trace")
    
    

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
    
    
    
    #print(TPS)

    #Energy Metrics    
    number_of_laps = 22
    # Interpolate wheel torque
    torque_func = interp1d(veh.vehicle_speed, veh.wheel_torque,kind='linear', fill_value='extrapolate')
    #print(V)
    wheel_torque = TPS * torque_func(V)
    motor_torque = wheel_torque / (veh.ratio_primary*veh.ratio_gearbox*veh.ratio_final*veh.n_primary*veh.n_gearbox*veh.n_final)
    
    #Power regen
    brake_force = BPS/veh.beta                          # F_brake = BPS/veh.beta
    overall_regen_efficiency = 0.9                      # assumes xx% of energy from braking is regenerated by the motor
    regen_power_limit = 20                              # maximum rate of regen in kW
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

    wheel_speed = V/veh.tyre_radius * 60/(2*np.pi)
    motor_speed = wheel_speed*veh.ratio_primary*veh.ratio_gearbox*veh.ratio_final

    #Work from the motor is force from the motor * dx integrated over the track
    total_work = np.trapz(wheel_torque / veh.tyre_radius, x = tr.x)
    
    #motor_power = motor_torque * motor_speed #in W
    #motor_energy_consumed = np.trapz(motor_power, x = time)    #in Joules



    energy_cost_total = number_of_laps*total_work/(3.6*10**6)       # in kWh
    energy_gained_total = number_of_laps*regenerated_energy/(3.6*10**6)   # in kWh
    
    #print("Energy cost is {:.3f} kWh".format(energy_cost_total)) 
    #print("Regenerated Power is {:.3f} kWh".format(energy_gained_total)) 
    #print()
    
    #veh.plotMotorCurve()

    return laptime, energy_cost_total




#simulate()

