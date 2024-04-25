'''
Rudimentary version of OpenAll for the custom point mass model

'''
import vehicle
import open_loop
import numpy as np
import pandas as pd
import track
import accumulator 

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import lap_utils
    
def read_info(workbook_file, sheet_name=1, start_row=2, end_row=1000, cols="A:B"):
    # Setup the Import Options
    opts = pd.io.excel.read_excel(workbook_file, sheet_name, header=None, skiprows=start_row-1, nrows=end_row-start_row+1, usecols=cols)
    
    # Specify column names
    opts.columns = ["Variable", "Value"]
    
    return opts

# estimate the number of points we will get with the car
def points_estimate(numLaps, time_endurance, time_autoX, time_acceleration, energy_endurance): 
    # endurance time and energy are for the whole race (22 laps)
    # autoX and acceleration times are for 1 run

    values = [1473.68, 
               46.85, 
               3.65, 
               1.75, 
               0.243, 
               0.841] 


    # Reading reference values from ptsRef file
    minimum_endurance_time = values[0]
    minimum_autoX_time = values[1]
    minimum_acceleration_time = values[2]
    minimum_energy_endurance = values[3]
    efficiency_factor_minimum = values[4]       # not used in previous calcs either
    efficiency_factor_maximum = values[5]

    # Adjust endurance time and energy for number of laps
    #tEnd = time_endurance*numLaps      # already taken care of
    #eEnd = energy_endurance*numLaps    # already taken care of

    max_endurance_time = 1.45*minimum_endurance_time

    # Calculate points for Endurance, Autocross, and Acceleration (from rulebook)
    ptsEnd = min(250, 250*((max_endurance_time/time_endurance)-1)/((max_endurance_time/minimum_endurance_time)-1))
    ptsAutoX = min(125, 118.5*((minimum_autoX_time*1.45/time_autoX)-1)/((minimum_autoX_time*1.45/minimum_autoX_time)-1) + 6.5)
    ptsAcc = min(100, 95.5*((minimum_acceleration_time*1.5/time_acceleration)-1)/((minimum_acceleration_time*1.5/minimum_acceleration_time)-1) + 4.5)

    # Calculate efficiency factor and points
    effFactor = (minimum_endurance_time/time_endurance)*(minimum_energy_endurance/energy_endurance)
    ptsEff = 0.0 #min(100, 100*(effFactor)/(efficiency_factor_maximum))

    
    # Return calculated points
    pts = [ptsEnd, ptsAutoX, ptsEff, ptsAcc, 0]

    # Dynamic points:
    dynamic_pts = np.sum(pts[:3])

    ptsTot = np.sum(pts)

    return (ptsEnd, ptsAutoX, ptsTot)


def mass_power_sweep():    

    # used for points estimation
    ptsRef_filename = 'SN3_Points_Reference.xlsx'
    numLaps = 22

    # loop over various motor_curves
    power_caps = [80, 70, 60, 50, 40, 30, 20, 10]

    # loop over various masses
    masses_to_sample = 50
    masses = np.linspace(200, 300, masses_to_sample)

    # output vectors of the sim for easier output in excel 
    end_times = []
    end_Es = []
    autoX_times = []
    autoX_Es = []
    accel_times = []
    caps = []
    Ms = []
    p1 = [] # points for end, autoX, and total
    p2 = []
    p3 = []
    

    # for graphing
    end_energies = np.zeros((len(masses), len(power_caps)))
    end_laptimes = np.zeros((len(masses), len(power_caps)))

    autoX_energies = np.zeros((len(masses), len(power_caps)))
    autoX_laptimes = np.zeros((len(masses), len(power_caps)))
    
    accel_laptimes = np.zeros((len(masses), len(power_caps)))

    end_points = np.zeros((len(masses), len(power_caps)))
    autoX_points = np.zeros((len(masses), len(power_caps)))
    tot_points = np.zeros((len(masses), len(power_caps)))

    #start endurance
    track.reload('Michigan 2014.xlsx')

    for i, new_mass in enumerate(masses):
        for j, power_cap in enumerate(power_caps):
            filename = "motor_curve{}.xlsx".format(power_cap)

            base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\motor_curves\\"
            filename = base_name + filename

            vehicle.soft_reload(new_mass, filename) # reload our vehicle to have the new data

            laptime, energy = open_loop.simulate()
            end_laptimes[i, j] = numLaps*laptime #3D grapher only likes 2D arrays; scale up by numLaps to get whole-race values
            end_energies[i, j] = numLaps*energy

            end_times.append(laptime)
            end_Es.append(energy)
            caps.append(power_cap)
            Ms.append(new_mass)

    
    # start AutoX
    track.reload('Michigan_2022_AutoX.xlsx')
    for i, new_mass in enumerate(masses):
            for j, power_cap in enumerate(power_caps):
                filename = "motor_curve{}.xlsx".format(power_cap)

                base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\motor_curves\\"
                filename = base_name + filename

                vehicle.soft_reload(new_mass, filename) # reload our vehicle to have the new data

                laptime, energy = open_loop.simulate()
                autoX_laptimes[i, j] = laptime #3D grapher only likes 2D arrays
                autoX_energies[i, j] = energy

                autoX_times.append(laptime)
                autoX_Es.append(energy)

                # account for 75m acceleration event 
                accel_time = lap_utils.simulate_accel(vehicle)
                accel_times.append(accel_time)
                accel_laptimes[i, j] = accel_time

                # estimate the points we get with this configuration
                points = points_estimate(numLaps, end_laptimes[i, j], laptime, accel_time, end_energies[i, j]) #no accel for now
                end_points[i, j] = points[0]
                autoX_points[i, j] = points[1]
                tot_points[i, j] = points[2]
                p1.append(points[0])
                p2.append(points[1])
                p3.append(points[2])


        
    # Output everything using Pandas
    header = ['Mass (kg)', 'Power Cap (kW)', 'Endurance Laptime (s)', 'Endurance Energy (kWh)', 'Endurance Points', 
              'AutoX Laptime (s)', 'AutoX Energy (kWh)', 'AutoX Points', 'Accel Time', 'Total Points']

    m = np.vstack(Ms)
    cap = np.vstack(caps)
    t1 = np.vstack(end_times)
    e1 = np.vstack(end_Es)
    t2 = np.vstack(autoX_times)
    e2 = np.vstack(autoX_Es)
    t3 = np.vstack(accel_times)
    p1 = np.vstack(p1)
    p2 = np.vstack(p2)
    p3 = np.vstack(p3)


    df0 = np.concatenate((m, cap, t1, e1, p1, t2, e2, p2, t3, p3), axis=1)
    df1 = pd.DataFrame(df0)
    writer = pd.ExcelWriter('Sweep_Data_6.xlsx', engine='xlsxwriter')

    df1.to_excel(writer, sheet_name='Sheet1', index=False, header=header)
    writer.close()
    
    #plot everything 

    fig = plt.figure()

    Ms_grid, caps_grid = np.meshgrid(masses, power_caps)
    
    # syntax for 3-D projection
    ax1 = fig.add_subplot(231, projection ='3d')
    ax2 = fig.add_subplot(232, projection ='3d')
    ax3 = fig.add_subplot(233, projection ='3d')
    ax4 = fig.add_subplot(234, projection ='3d')
    ax5 = fig.add_subplot(235, projection ='3d')
    ax6 = fig.add_subplot(236, projection ='3d')


    # plotting
    surf = ax1.plot_surface(Ms_grid, caps_grid, end_laptimes.T, cmap='cool', alpha=0.8, 
                           facecolors=plt.cm.cool(end_energies.T / np.max(end_energies))) #add energy as a color metric
    # colorbar set up
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool', norm=plt.Normalize(np.min(end_energies), np.max(end_energies))), ax=ax1)
    cbar.set_label('Energy (kWh)')

    ax1.set_title('Endurance Laptimes and Energies')
    ax1.set_xlabel('Masses (kg)', fontsize=12)
    ax1.set_ylabel('Power Caps (kW)', fontsize=12)
    ax1.set_zlabel('Laptimes (s)', fontsize=12)

    surf = ax2.plot_surface(Ms_grid, caps_grid, autoX_laptimes.T, cmap='cool', alpha=0.8, 
                           facecolors=plt.cm.cool(autoX_energies.T / np.max(autoX_energies))) #add energy as a color metric
    # colorbar set up
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool', norm=plt.Normalize(np.min(autoX_energies), np.max(autoX_energies))), ax=ax2)
    cbar.set_label('Energy (kWh)')

    ax2.set_title('AutoX Laptimes and Energies')
    ax2.set_xlabel('Masses (kg)', fontsize=12)
    ax2.set_ylabel('Power Caps (kW)', fontsize=12)
    ax2.set_zlabel('Laptimes (s)', fontsize=12)

    surf = ax3.plot_surface(Ms_grid, caps_grid, tot_points.T, alpha=0.8) 
    ax3.set_title('Total Points')
    ax3.set_xlabel('Masses (kg)', fontsize=12)
    ax3.set_ylabel('Power Caps (kW)', fontsize=12)
    ax3.set_zlabel('Points', fontsize=12)


    surf = ax4.plot_surface(Ms_grid, caps_grid, end_points.T, alpha=0.8) 
    ax4.set_title('Endurance Points')
    ax4.set_xlabel('Masses (kg)', fontsize=12)
    ax4.set_ylabel('Power Caps (kW)', fontsize=12)
    ax4.set_zlabel('Points', fontsize=12)

    surf = ax5.plot_surface(Ms_grid, caps_grid, autoX_points.T, alpha=0.8) 
    ax5.set_title('AutoX Points')
    ax5.set_xlabel('Masses (kg)', fontsize=12)
    ax5.set_ylabel('Power Caps (kW)', fontsize=12)
    ax5.set_zlabel('Points', fontsize=12)

    surf = ax6.plot_surface(Ms_grid, caps_grid, accel_laptimes.T, alpha=0.8) 
    ax6.set_title('Accel Times (75 m)')
    ax6.set_xlabel('Masses (kg)', fontsize=12)
    ax6.set_ylabel('Power Caps (kW)', fontsize=12)
    ax6.set_zlabel('Times (s)', fontsize=12)

    plt.show()
     
# attempts to drain the accumulator for a given number of laps given pack data 
def try_drainage(pack_data, motor_power, V, laps_to_try, starting_discharge=0):
    # Accumulator sizing calculations
    dt = np.divide(track.dx, V)
    adjusted_velocities = V
    total_laptime = 0             # total laptime due to field-weakening adjustments

    # set up our accumulator
    pack = accumulator.Pack()
    pack.pack(pack_data[0], pack_data[1], pack_data[2]) 
    pack.set_discharge(starting_discharge)                      # start with some energy depleted

    # flag to check if we over-drain our accumulator
    lapFailure = -1      
    for j in range(laps_to_try):
        # assume the pack is not drained this lap (discharge still carries over)
        pack.set_drain_error(False)

        # modify the laptime due to field-weakening
        for i, power in enumerate(motor_power):
            # drain the accumulator by the energy consumed during each time-step; smoothing included
            pack.new_drain(power, dt[i])

        # calculate the accumulator voltage and base speed at this point 
        cell_data = pack.get_cell_data()
        accumulator_voltage = cell_data["voltage"]
        base_speed = 60 * accumulator_voltage / (20*np.pi/10.12)
            
        # to account for field-weakening, we cut torque to any rpm above the base speed
        # this makes base_speed effectively 90% the max speed of the car
        threshold = 1/0.9 

        adjusted_velocities = np.where(adjusted_velocities > threshold*base_speed, threshold*base_speed, adjusted_velocities)
            
        # assume our dts don't change by much; in the future, we can iterate on this until we get decent convergence
        updated_dt = np.divide(track.dx, adjusted_velocities)

        total_laptime += np.cumsum(updated_dt)[-1]

        # if we fail midway through a race and want to drop power caps, we need to re-do the lap at the lower power cap
        if pack.is_depleted():
            lapFailure = j
            print("Drain Failure with discharge {:.2f} of {:.2f}".format(cell_data["discharge"], cell_data["capacity"]))

            break
    
    # update the laptime to account for base speed field weakening
    return lapFailure, total_laptime, pack.get_cell_data()["discharge"]

def optimize_accumulator():
    #accumulator sweep
    possible_packs = [[14, 4, 10]] #[[16, 5, 7], [16, 4, 8]] #]  

    '''[[12, 4, 9], [12, 4, 10], [12, 5, 8], [12, 5, 9],
                      [12, 5, 10], [13, 4, 9], [13, 4, 10], [13, 5, 7], 
                      [13, 5, 8], [13, 5, 9], [14, 4, 8], [14, 4, 9], 
                      [14, 4, 10], [14, 5, 7], [14, 5, 8], [14, 5, 9], 
                      [15, 4, 8], [15, 4, 9], [15, 5, 6], [15, 5, 7], 
                      [15, 5, 8], [16, 4, 7], [16, 4, 8], [16, 5, 6], 
                      [16, 5, 7], [16, 5, 8]] '''
    
    
    # used for points estimation
    ptsRef_filename = 'SN3_Points_Reference.xlsx'
    numLaps = 22

    # loop over various motor_curves
    power_caps = [60, 50, 40, 30, 20] #[80, 70, 60, , 10] #10 is unrealistic, removed for speed

    #car mass in kg, no cells, with driver weight 80kg (from SN3 mass spec sheet)
    base_mass = 172.1+80                 
    masses = []
    pack_names = []
    capacities = []

    for pack_dimension in possible_packs:
        series, parallel, segment = pack_dimension
        pack = accumulator.Pack()
        pack.pack(series, parallel, segment)
        cell_data = pack.get_cell_data()
        cell_mass = cell_data["weight"]/1000    #in kg

        masses.append(cell_mass+base_mass)
        pack_names.append(cell_data["name"])
        capacities.append(cell_data["capacity"]/1000)       # in kWh
         

    #start endurance
    track.reload('Michigan_2021_Endurance.xlsx')

    # output vectors of the sim for easier output in excel 
    end_times = []
    end_Es = []
    power_cap1 = []
    power_cap2 = []
    switch_point = []
    


    # loop over all packs and try to find the fastest laptime that still completes
    for i, new_mass in enumerate(masses):        
        # index of power caps at each point
        j1 = 0
        j2 = 3 #TEMP
        dropped_laps = 0            # how many laps do we complete at a lower power cap
        race_unfinished = True
        
        second_failure = -1
        safety_breaker = 0
        
        while race_unfinished:
            # run a simulation with new_mass and the high power cap
            filename = "motor_curve{}.xlsx".format(power_caps[j1])
            base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\motor_curves\\"
            filename = base_name + filename
            vehicle.soft_reload(new_mass, filename) # reload our vehicle to have the new data

            laptime_first, energy_first, motor_power_first, V_first = open_loop.simulate()

            # run a simulation with new_mass and the low power cap
            filename = "motor_curve{}.xlsx".format(power_caps[j2])
            base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\motor_curves\\"
            filename = base_name + filename
            vehicle.soft_reload(new_mass, filename) # reload our vehicle to have the new data

            laptime_second, energy_second, motor_power_second, V_second = open_loop.simulate()
            
            # unless we switch, there is no 2nd laptime
            laptime_second = 0.0
            discharge2 = 0.0

            # initially, try the whole race at the max power cap; otherwise, drop the power cap
            first_failure, laptime_first, discharge = try_drainage(possible_packs[i], motor_power_first, V_first, numLaps-dropped_laps)
            if dropped_laps > 0:
                second_failure, laptime_second, discharge2 = try_drainage(possible_packs[i], motor_power_second, V_second, dropped_laps, discharge)

            # Drop the power cap one lap before we fail, unless the second leg also fails  
            # this should always trigger on the first pass through if we fail
            if (first_failure >= 0 and second_failure < 0):
                dropped_laps = numLaps - first_failure + 1

            # if we fail on the second leg, we need to drop power cap sooner
            elif (second_failure >= 0 and dropped_laps < numLaps-2):
                dropped_laps += 1
            
            # if we still fail after dropping at the start, just drop more
            elif (second_failure >= 0 or first_failure >= 0):
                print("Decreasing power cap to {}".format(power_caps[j1+1]))
                j1 += 1
                j2 += 1
                dropped_laps = 0
                second_failure = -1

                safety_breaker += 1

                if safety_breaker > 100:
                    # unable to converge; failure
                    print("Unable to converge")
                    j1 = -1
                    j2 = -1
                    dropped_laps = 0
                    break

            # if both legs succeed, we finish the race
            else:
                race_unfinished = False

            total_laptime = laptime_first+laptime_second
            print("First Laptime: {}, Second: {}".format(laptime_first, laptime_second))
            print("First Discharge: {}, Second: {}".format(discharge, discharge2))
            print("First Power Cap: {}, Second: {}".format(power_caps[j1], power_caps[j2]))
            print("First Failure: {}, Second: {}; dropped laps {}".format(first_failure, second_failure, dropped_laps))

            #TODO we often miss by 2, even when we drop power caps, even when we drop earlier. Consider dropping by 2 power caps in this case
            total_energy = energy_first*(numLaps-dropped_laps)+energy_second*dropped_laps

        # output the optimal laptime / energy and power cap scheme for each pack config
        end_times.append(total_laptime)
        end_Es.append(total_energy)
        power_cap1.append(power_caps[j1])
        power_cap2.append(power_caps[j2])
        switch_point.append(numLaps-dropped_laps)
        


        
    # Output everything using Pandas
    header = ['Pack Config', 'Mass (kg)', 'Capacity (kWh)', 'Power Cap 1 (kW)', 'Power Cap 2 (kW)', 
              'Switch Point', 'Endurance Laptime (s)', 'Endurance Energy (kWh)']

    df0 = np.concatenate((
    np.vstack(pack_names), 
    np.vstack(masses), 
    np.vstack(capacities),
    np.vstack(power_cap1), 
    np.vstack(power_cap2), 
    np.vstack(switch_point), 
    np.vstack(end_times), 
    np.vstack(end_Es)), axis=1)


    df1 = pd.DataFrame(df0)
    writer = pd.ExcelWriter('accumulator_sims5.xlsx', engine='xlsxwriter')

    df1.to_excel(writer, sheet_name='Sheet1', index=False, header=header)
    writer.close()
    
# determines laptime and consumption for 4 runs of autoX, followed by 4 of accel 
# if unable to finish, drops the power cap and searches for the best solution       
def wendys_four_by_four(pack_data):
    # set up our accumulator
    pack = accumulator.Pack()
    pack.pack(pack_data[0], pack_data[1], pack_data[2]) 

    #car mass in kg, no cells, with driver weight 80kg (from SN3 mass spec sheet)
    base_mass = 172.1+80                 

    cell_mass = pack.get_cell_data()["weight"]/1000    #in kg

    total_car_mass = cell_mass+base_mass
    pack_name = pack.get_cell_data()["name"]

    # set up 4 runs of autoX
    track.reload('Michigan_2022_AutoX.xlsx')
    numLaps = 4
    power_caps = [80, 70, 60, 50]
    

    # index of power caps at each point
    j1 = 0
    j2 = 1
    dropped_laps = 0            # how many laps do we complete at a lower power cap
    race_unfinished = True
    total_laptime = 0
    total_energy = 0

    second_failure = -1
    while race_unfinished:
        # run a simulation with new_mass and the high power cap
        filename = "motor_curve{}.xlsx".format(power_caps[j1])
        base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\motor_curves\\"
        filename = base_name + filename
        vehicle.soft_reload(total_car_mass, filename) # reload our vehicle to have the new data

        laptime_first, energy_first, motor_power_first, V_first = open_loop.simulate()

        # run a simulation with new_mass and the low power cap
        filename = "motor_curve{}.xlsx".format(power_caps[j2])
        base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\motor_curves\\"
        filename = base_name + filename
        vehicle.soft_reload(total_car_mass, filename) # reload our vehicle to have the new data

        laptime_second, energy_second, motor_power_second, V_second = open_loop.simulate()
        
        # unless we switch, there is no 2nd laptime
        laptime_second = 0.0
        
        # initially, try the whole race at the max power cap; otherwise, drop the power cap
        first_failure, laptime_first, discharge = try_drainage(pack_data, motor_power_first, V_first, numLaps-dropped_laps)
        if dropped_laps > 0:
            second_failure, laptime_second, discharge2 = try_drainage(pack_data, motor_power_second, V_second, dropped_laps, discharge)

        # if we cannot complete the race, drop the power cap sooner
        if ((first_failure > 1 or second_failure > 1) and dropped_laps < numLaps-2):
            print("Dropping power cap for another lap")
            dropped_laps += 1
        
        # if we drop for more than numLaps-2 laps, just drop the power cap and reset
        elif (first_failure > 1 or second_failure > 1):
            print("Decreasing power cap to {}".format(power_caps[j1+1]))
            j1 += 1
            j2 += 1
            dropped_laps = 0
            second_failure = -1

        # if we just can't finish at all, quit
        elif ((second_failure > 1) and dropped_laps > 20):
            # unable to converge; failure
            print("Unable to converge")
            j1 = -1
            j2 = -1
            dropped_laps = 0
            break

        # if both legs succeed, we finish the race
        else:
            race_unfinished = False

        autoX_laptime = laptime_first+laptime_second
        autoX_energy = energy_first*(numLaps-dropped_laps)+energy_second*dropped_laps


    # account for 75m acceleration event 
    accel_time, final_speed = lap_utils.simulate_accel(vehicle)
    # approximate energy consumption as 1/2 m v^2 (assume 100% efficiency) for small consumption
    accel_consumption = 0.5*total_car_mass * final_speed**2
    accel_consumption *= numLaps/(3.6*10**6)                        # all 4 laps, in kWh

    # returns: the total laptime for all 4 laps of autoX, all energy consumed in kWh, 
    # total time for a single 75m accel, total energy consumed during accel
    # index of first power cap, index of second power cap, number of laps run at second power cap
    return autoX_laptime, autoX_energy, accel_time, accel_consumption, j1, j2, dropped_laps

# determines laptime and consumption for 4 runs of autoX, followed by 4 of accel        
def four_by_four_simple(pack_data):
    # set up our accumulator
    pack = accumulator.Pack()
    pack.pack(pack_data[0], pack_data[1], pack_data[2]) 

    #car mass in kg, no cells, with driver weight 80kg (from SN3 mass spec sheet)
    base_mass = 172.1+80                 

    cell_mass = pack.get_cell_data()["weight"]/1000    #in kg

    total_car_mass = cell_mass+base_mass
    pack_name = pack.get_cell_data()["name"]

    # set up 4 runs of autoX
    track.reload('Michigan_2022_AutoX.xlsx')
    numLaps = 4
    power_cap = 80
    
    filename = "motor_curve{}.xlsx".format(power_cap)
    base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\motor_curves\\"
    filename = base_name + filename
    vehicle.soft_reload(total_car_mass, filename) # reload our vehicle to have the new data

    one_lap_laptime, one_lap_energy, motor_power, V = open_loop.simulate()

    first_failure, autoX_laptime, discharge = try_drainage(pack_data, motor_power, V, numLaps)
    autoX_energy = one_lap_energy * numLaps

    # account for 75m acceleration event 
    accel_time, final_speed = lap_utils.simulate_accel(vehicle)
    # approximate energy consumption as 1/2 m v^2 (assume 100% efficiency) for small consumption
    accel_consumption = 0.5*total_car_mass * final_speed**2         # single lap, in J
    accel_consumption *= numLaps/(3.6*10**6)                        # 4 laps, in kWh

    return autoX_laptime, autoX_energy, accel_time, accel_consumption, first_failure


    

# use our four_by_four method to get nice data
def four_by_four_test():
    possible_packs = [[16, 5, 7], [16, 4, 8], [14, 4, 10]]
    power_caps = [80, 70, 60, 50]

    configs = []
    masses = []
    capacities = []
    autoX_times = []
    autoX_energies = []
    accel_times = []
    accel_consumptions = []
    power_cap1 = []
    power_cap2 = []
    dropped_laps = []

    for pack_data in possible_packs:
        # set up our accumulator
        pack = accumulator.Pack()
        pack.pack(pack_data[0], pack_data[1], pack_data[2]) 

        #car mass in kg, no cells, with driver weight 80kg (from SN3 mass spec sheet)
        base_mass = 172.1+80                 

        cell_mass = pack.get_cell_data()["weight"]/1000    #in kg

        total_car_mass = cell_mass+base_mass
        pack_name = pack.get_cell_data()["name"]

        autoX_laptime, autoX_energy, accel_time, accel_consumption, j1, j2, dropped_lap = wendys_four_by_four(pack_data)
        
        configs.append(pack_name)
        masses.append(total_car_mass)
        capacities.append(pack.get_cell_data()["capacity"]/1000)    #in kWh
        autoX_times.append(autoX_laptime)
        autoX_energies.append(autoX_energy)
        accel_times.append(accel_time)
        accel_consumptions.append(accel_consumption)
        power_cap1.append(power_caps[j1])
        power_cap2.append(power_caps[j2])
        dropped_laps.append(dropped_lap)


    # Output everything using Pandas
    header = ['Pack Config', 'Mass (kg)', 'Capacity (kWh)', 'Power Cap 1 (kW)', 'Power Cap 2 (kW)', 
              'Dropped Laps', 'Total AutoX Laptime (s)', 'Total AutoX Energy (kWh)', 'Accel Laptime (s)', 'Total Accel Energy (kWh)']

    df0 = np.concatenate((
    np.vstack(configs), 
    np.vstack(masses), 
    np.vstack(capacities),
    np.vstack(power_cap1), 
    np.vstack(power_cap2), 
    np.vstack(dropped_laps),
    np.vstack(autoX_times), 
    np.vstack(autoX_energies), 
    np.vstack(accel_times), 
    np.vstack(accel_consumptions)), axis=1)


    df1 = pd.DataFrame(df0)
    writer = pd.ExcelWriter('four_by_four1.xlsx', engine='xlsxwriter')

    df1.to_excel(writer, sheet_name='Sheet1', index=False, header=header)
    writer.close()

    return df1

# use the data from optimize_accumulator and four_by_four_test() to estimate points for the pack
# make sure the rows are properly --i.e. consistently-- aligned in both files
def accumulator_points(): 
    # import our endurance and autoX data
    four_by_four_filename = 'four_by_four1.xlsx'
    endurance_filename = 'accumulator_sims3.xlsx'

    four_data = pd.io.excel.read_excel(four_by_four_filename, nrows=100)
    #four_data.columns = ['Pack Config', 'Mass (kg)', 'Capacity (kWh)', 'Power Cap 1 (kW)', 'Power Cap 2 (kW)', 
    # 'Dropped Laps', 'Total AutoX Laptime (s)', 'Total AutoX Energy (kWh)', 'Accel Laptime (s)', 'Total Accel Energy (kWh)']
    
    endurance_data = pd.io.excel.read_excel(endurance_filename, nrows=100)
    #endurance_data.columns = ['Pack Config', 'Mass (kg)', 'Capacity (kWh)', 'Power Cap 1 (kW)', 'Power Cap 2 (kW)', 
    #     'Switch Point', 'Endurance Laptime (s)', 'Endurance Energy (kWh)']
    

    numLaps = 22

    endurance_times = endurance_data.loc[:, "Endurance Laptime (s)"]
    endurance_energies = endurance_data.loc[:, "Endurance Energy (kWh)"]

    autoX_times = four_data.loc[:, "Total AutoX Laptime (s)"] / 4         # single-lap laptime; we have 4-lap laptime
    accel_times = four_data.loc[:, "Accel Laptime (s)"]

    points = []
    for i in range(len(endurance_times)):
        points.append(points_estimate(numLaps, endurance_times[i], autoX_times[i], accel_times[i], endurance_energies[i]))

    return points

#four_by_four_test()
optimize_accumulator()

#print(accumulator_points())


