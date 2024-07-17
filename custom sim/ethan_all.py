'''
Functions to process results output from open_loop.
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

# predict our efficiency factor and score from the average laptime and energy drain
def calculate_efficiency_factor(avg_laptime, energy_drain):
    # assume LapTotal min, yours, and CO2 are all the same (22)
    t_min = 67.667
    t_max = 98.118
    co2_min = 0.0518                            # avg kg CO2 per lap
    eff_factor_min = 0.0591
    eff_factor_max = 0.841

    
    # predict RIT's efficiency factor
    '''
    t_yours = 73.467
    e_yours = 4.903                             # kWh
    co2_yours = e_yours * 0.65
    co2_yours = co2_yours / 22                  # avg adjusted kg CO2 per lap
    RIT_factor = t_min/t_yours * co2_min / co2_yours
    RIT_score = 100 * (RIT_factor-eff_factor_min)/(eff_factor_max-eff_factor_min)

    print(RIT_factor)
    print(RIT_score)
    print(eff_factor_min)
    '''
    
    # predict our efficiency factor
    co2_yours = energy_drain * 0.65
    co2_yours = co2_yours / 22                  # avg adjusted kg CO2 per lap
    eff_factor = t_min/avg_laptime * co2_min / co2_yours


    #linear score approximator from eff_factor
    m = (100-81.3)/(0.841-0.243)
    Points = 89.9+m*(eff_factor-0.362)
    #print(Points)

    return eff_factor, Points


# estimate the number of points we will get with the car
def points_estimate(numLaps, time_endurance, energy_endurance, time_autoX, time_acceleration): 
    # endurance time and energy are for the whole race (22 laps)
    # autoX and acceleration times are for 1 run

    reference_values = [1473.68, 
                        46.85, 
                        3.65, 
                        1.75, 
                        0.243, 
                        0.841] 


    # Reading reference values from ptsRef file
    minimum_endurance_time = reference_values[0]
    minimum_autoX_time = reference_values[1]
    minimum_acceleration_time = reference_values[2]
    
    max_endurance_time = 1.45*minimum_endurance_time
    max_autoX_time = 1.45*minimum_autoX_time
    max_accel_time = 1.5*minimum_acceleration_time

    # Calculate points for Endurance, Autocross, and Acceleration (from rulebook)
    ptsEnd = min(250, 250*((max_endurance_time/time_endurance)-1)/((max_endurance_time/minimum_endurance_time)-1))
    ptsAutoX = min(125, 118.5*((max_autoX_time/time_autoX)-1)/((max_autoX_time/minimum_autoX_time)-1) + 6.5)
    ptsAcc = min(100, 95.5*((max_accel_time/time_acceleration)-1)/((max_accel_time/minimum_acceleration_time)-1) + 4.5)


    # Calculate efficiency factor and points
    CO2_conversion = 0.65 #kg CO2 per kWh for electric
    RIT_scale_factor = 0.590709    # convert from our estimated energy to the "as-read" energy from the sensor

    # hardcoded results from 2023
    CO2_min = 0.0518
    minimum_endurance_time = 67.667
    eff_factor_min = 0.059
    eff_factor_max = 0.841


    # scale factor to match our computed energy with the as-read energy from RIT
    

    # estimate our efficiency factor from 2023 results 
    #CO2_ours = CO2_conversion*energy_endurance*RIT_scale_factor/numLaps
    #efficiency_factor = (minimum_endurance_time/time_endurance) * (CO2_min/CO2_ours)
    efficiency_factor, efficiency_score_2023 = calculate_efficiency_factor(time_endurance/numLaps, energy_endurance)

    # using the 2023 metric to calculate efficiency
    #efficiency_score_2023 = min(100, 100*((eff_factor_min/efficiency_factor)-1)/((eff_factor_min/eff_factor_max)-1))

    # using the proposed 2024 metric
    efficiency_score_linear = min(100, 100*(efficiency_factor-eff_factor_min)/(eff_factor_max-eff_factor_min))

    ptsEff = efficiency_score_linear

    # Return calculated points
    pts = [ptsEnd, ptsAutoX, 0.0, ptsEff, 0.0]

    ptsTot = np.sum(pts)

    return (ptsEnd, ptsAutoX, ptsEff, ptsTot)

# alter this as needed, depending on what we want to sweep
def autoX_sweep():    
    # used for points estimation
    ptsRef_filename = 'SN3_Points_Reference.xlsx'
    numLaps = 22

    # loop over various motor_curves
    power_caps = [24, 20, 15]

    # Cl, Cd
    aero_coefficients = [[-1.98, -1.33],        # High downforce config
                         [-1.23, -0.93],        # Low drag config
                         [-0.05, -0.52]]        # No aero package

    base_mass = 172.1+80    

    # with aero, no driver;     245.0
    # without aero, no driver;  231.0

    # driver weight:        80


    pack_dimension = [14, 4, 10]

    # initialize our pack
    series, parallel, segment = pack_dimension
    pack = accumulator.Pack()
    pack.pack(series, parallel, segment)

    # get the cell mass, name, and capacity for output
    cell_data = pack.get_cell_data()
    capacity = cell_data["capacity"]/1000       # in kWh

    capacities = []

    driver_mass = 87
    base_mass = 231.0 + driver_mass
    with_aero = 245.0 + driver_mass         # for high downforce config

    masses = [with_aero, with_aero, base_mass]
    
    # output vectors of the sim for easier output in excel 
    autoX_times = []
    Es = []
    caps = []
    Ms = []
    Cls = []
    Cds = []
    accel_times = []

    #start endurance
    track.reload('Michigan_2022_AutoX.xlsx')

    for i, aero_package in enumerate(aero_coefficients):
        for j, power_cap in enumerate(power_caps):
            pack.reset()
            vehicle.soft_reload(masses[i], power_cap, aero_package) # reload our vehicle to have the new data

            laptime, energy, transient_output = open_loop.simulate(pack)

            pack_failed = pack.is_depleted()

            if not (pack_failed):
                autoX_times.append(laptime)
                Es.append(energy)
                caps.append(power_cap)
                Ms.append(masses[i])
                Cls.append(aero_package[0])
                Cds.append(aero_package[1])
                capacities.append(capacity)

                # Run accel 
                accumulator_voltage = cell_data["voltage"]
                accel_time, final_velocity = lap_utils.simulate_accel(vehicle, accumulator_voltage)
                accel_times.append(accel_time)

                writer = pd.ExcelWriter('transient_output_pack{:}_{:}kW.xlsx'.format(i, power_cap), engine='xlsxwriter')

                transient_output.to_excel(writer, sheet_name='Sheet1', index=False)
                writer.close()


            else:
                print("Pack failure at {}".format(power_cap))


    



    # Output everything using Pandas
    header = ['Mass (kg)', 'Capacity (kWh)', 'Power Cap (kW)', 'Cls', 'Cds', 'AutoX Laptime (s)', 'AutoX Energy (kWh)', 'Accel Laptime (s)']

    m = np.vstack(Ms)
    Capacities = np.vstack(capacities)
    cap = np.vstack(caps)
    clift = np.vstack(Cls)
    cdrag = np.vstack(Cds)
    t1 = np.vstack(autoX_times)
    e1 = np.vstack(Es)
    acc = np.vstack(accel_times)
  

    df0 = np.concatenate((m, Capacities, cap, clift, cdrag, t1, e1, acc), axis=1)
    df1 = pd.DataFrame(df0)
    writer = pd.ExcelWriter('autoX_data2.xlsx', engine='xlsxwriter')

    df1.to_excel(writer, sheet_name='Sheet1', index=False, header=header)
    writer.close()

# alter this as needed, depending on what we want to sweep
def aero_sweep():    
    # used for points estimation
    ptsRef_filename = 'SN3_Points_Reference.xlsx'
    numLaps = 22

    # loop over various motor_curves
    power_caps = [30, 24, 20, 15]

    # Cl, Cd
    aero_coefficients = [#[-1.98, -1.33],            # High downforce config; negative = downforce, + is lift
                        [-1.23, -0.93]]#,           # Low drag config; must be negative!
                        #[-0.05, -0.52]]           # No aero package


    # with aero, no driver;     245.0
    # without aero, no driver;  231.0

    # driver weight:        80


    pack_dimension = [14, 4, 10]

    # initialize our pack
    series, parallel, segment = pack_dimension
    pack = accumulator.Pack()
    pack.pack(series, parallel, segment)

    # get the cell mass, name, and capacity for output
    cell_data = pack.get_cell_data()
    capacity = cell_data["capacity"]/1000       # in kWh

    capacities = []

    driver_masses = [45, 55, 65, 75, 90]
    #base_mass = 231.0 + driver_mass
    with_aero = 245.0          # for high downforce config

    masses = []

    for driver_mass in driver_masses:
        masses.append(with_aero + driver_mass)
    

    # output vectors of the sim for easier output in excel 
    end_times = []
    end_Es = []
    caps = []
    Ms = []
    Cls = []
    Cds = []

    #start endurance
    track.reload('Michigan_2021_Endurance.xlsx')

    for i, mass in enumerate(masses):
        for j, power_cap in enumerate(power_caps):

            aero_package = aero_coefficients[0]
            pack.reset()
            vehicle.soft_reload(masses[i], power_cap, aero_package) # reload our vehicle to have the new data

            laptime, energy, pack_failed = open_loop.simulate_endurance(pack, numLaps)

            if not (pack_failed):
                end_times.append(laptime)
                end_Es.append(energy)
                caps.append(power_cap)
                Ms.append(masses[i])
                Cls.append(aero_package[0])
                Cds.append(aero_package[1])
                capacities.append(capacity)

                
    
    # Output everything using Pandas
    header = ['Mass (kg)', 'Capacity (kWh)', 'Power Cap (kW)', 'Cls', 'Cds', 'Endurance Laptime (s)', 'Endurance Energy (kWh)']

    m = np.vstack(Ms)
    Capacities = np.vstack(capacities)
    cap = np.vstack(caps)
    clift = np.vstack(Cls)
    cdrag = np.vstack(Cds)
    t1 = np.vstack(end_times)
    e1 = np.vstack(end_Es)
  

    df0 = np.concatenate((m, Capacities, cap, clift, cdrag, t1, e1), axis=1)
    df1 = pd.DataFrame(df0)
    writer = pd.ExcelWriter('mass_power_sweep.xlsx', engine='xlsxwriter')

    df1.to_excel(writer, sheet_name='Sheet1', index=False, header=header)
    writer.close()
     

# Estimate points for previously run outputs of endurance, autoX, and accel
def points_from_spreadsheet():
    numEnduranceLaps = 22

    endurance_reference_file = "endurance_data6.xlsx"
    endurance_data = pd.io.excel.read_excel(endurance_reference_file, sheet_name=1)

    autoX_reference_file ="autoX_data1.xlsx"
    autoX_data = pd.io.excel.read_excel(autoX_reference_file, sheet_name=1)

        
    # Convert all our data to floats
    endurance_times = endurance_data.loc[:, "Endurance Laptime (s)"]
    endurance_times = endurance_times.astype(float)

    endurance_energies = endurance_data.loc[:, "Endurance Energy (kWh)"]
    endurance_energies = endurance_energies.astype(float)

    autoX_times = autoX_data.loc[:, "AutoX Laptime (s)"] 
    autoX_times = autoX_times.astype(float)

    accel_times = autoX_data.loc[:, "Accel Laptime (s)"]
    accel_times = accel_times.astype(float)

    end_points = []
    eff_points = []
    autoX_points = []
    total_points = []
    for i in range(len(endurance_times)):
        points = points_estimate(numEnduranceLaps, endurance_times[i], endurance_energies[i], autoX_times[i], accel_times[i])
        end_points.append(points[0])
        autoX_points.append(points[1])
        eff_points.append(points[2])
        total_points.append(points[3])

    points_header = ['Endurance Pts', 'AutoX Pts', 'Efficiency Pts', 'Total Pts']

    points_data = pd.DataFrame(data=zip(end_points, autoX_points, eff_points, total_points), columns=points_header)

    # Output all the data, inclusding the points
    writer = pd.ExcelWriter('aero_points_sims3.xlsx', engine='xlsxwriter')
    points_data.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()

def optimize_endurance(endurance_trackfile, possible_packs, power_caps, numLaps, header):
    
    # load our endurance track into the sim
    track.reload(endurance_trackfile)

    # set up our output vectors
    masses = []
    pack_names = []
    capacities = []        
    end_times = []
    end_Es = []
    power_cap1 = []
    power_cap2 = []
    dropped_laps = []

    #car mass in kg, no cells, with driver weight 80kg (from SN3 mass spec sheet)
    base_mass = 172.1+80    
    
    # loop over all packs and try to find the fastest laptime that still completes
    for i, pack_dimension in enumerate(possible_packs):      
        
        # initialize our pack
        series, parallel, segment = pack_dimension
        pack = accumulator.Pack()
        pack.pack(series, parallel, segment)

        # get the cell mass, name, and capacity for output
        cell_data = pack.get_cell_data()
        cell_mass = cell_data["weight"]/1000    #in kg

        masses.append(cell_mass+base_mass)
        pack_names.append(cell_data["name"])
        capacities.append(cell_data["capacity"]/1000)       # in kWh


        # index of power caps at each point
        j1 = 0
        race_unfinished = True
        
        # drop power caps until we can complete endurance without failure of this pack 
        while race_unfinished:
            # reset our pack before each test
            pack.reset()

            # initialize our vehicle with the new mass and capped motor curve 
            vehicle.soft_reload(masses[i], power_caps[j1]) # reload our vehicle to have the new data

            # simulate endurance
            laptime, energy, pack_failure = open_loop.simulate_endurance(pack, numLaps)

            # if we cannot complete endurance with this power cap, drop the cap
            if pack_failure:
                j1 += 1
            else:
                race_unfinished = False


        # interpolate between our solutions - > unnecessary for now

        # output the optimal laptime / energy and power cap scheme for each pack config
        end_times.append(laptime)
        end_Es.append(energy)
        power_cap1.append(power_caps[j1])        

    # verticalize and concatenate all our vectors
    df0 = np.concatenate((
    np.vstack(pack_names), 
    np.vstack(masses), 
    np.vstack(capacities),
    np.vstack(power_cap1), 
    np.vstack(end_times), 
    np.vstack(end_Es)), axis=1)

    df1 = pd.DataFrame(df0, columns=header)
    return df1
    
# use the old 4x4 if necessary, but consumption seemed negligible
def optimize_autoX(autoX_trackfile, possible_packs, power_caps, numLaps, autoX_header):
    # optimization is the same for endurance and autoX
    df2 = optimize_endurance(autoX_trackfile, possible_packs, power_caps, numLaps, autoX_header)

    # drop the first 3 columns for redundancy
    df2 = df2.drop(columns=['Pack Config', 'Mass (kg)', 'Capacity (kWh)'], axis=1)
    return df2

def optimize_accel(possible_packs, numLaps):
    accel_times = []
    accel_Es = []
    
    #car mass in kg, no cells, with driver weight 80kg (from SN3 mass spec sheet)
    base_mass = 172.1+80   
    masses = []
    
    for i, pack_dimension in enumerate(possible_packs):      
        # initialize our pack
        series, parallel, segment = pack_dimension
        pack = accumulator.Pack()
        pack.pack(series, parallel, segment)

        # get the cell mass, name, and capacity for output
        cell_data = pack.get_cell_data()
        cell_mass = cell_data["weight"]/1000    #in kg

        masses.append(cell_mass+base_mass)

        # account for 75m acceleration event 
        accel_time, final_speed = lap_utils.simulate_accel(vehicle)
        # approximate energy consumption as 1/2 m v^2 (assume 100% efficiency) for small consumption
        accel_consumption = 0.5*masses[i] * final_speed**2
        accel_consumption *= numLaps/(3.6*10**6)                        # all 4 laps, in kWh
        accel_times.append(accel_time)
        accel_Es.append(accel_consumption)


    df0 = np.concatenate((
    np.vstack(accel_times),
    np.vstack(accel_Es)), axis=1)

    df1 = pd.DataFrame(df0, columns = ['Accel Laptime (s)', 'Accel Energy (kWh)'])
    return df1 


def accumulator_points():
    ptsRef_filename = 'SN3_Points_Reference.xlsx'

    endurance_trackfile = 'Michigan_2021_Endurance.xlsx'
    autoX_trackfile = 'Michigan_2022_AutoX.xlsx'

    possible_packs = [[14, 4, 10], [16, 5, 7]] #, [16, 4, 8]] #, [14, 4, 10]]

    # Run endurance, autoX, and accel
    power_caps = [50, 30]
    numEnduranceLaps = 2
    numAutoXLaps = 1

    end_header = ['Pack Config', 'Mass (kg)', 'Capacity (kWh)', 'Endurance Power Cap (kW)', 
             'Endurance Laptime (s)', 'Endurance Energy (kWh)']
    
    autoX_header = ['Pack Config', 'Mass (kg)', 'Capacity (kWh)', 'AutoX Power Cap (kW)', 
             'AutoX Laptime (s)', 'AutoX Energy (kWh)']
    


    endurance_data = optimize_endurance(endurance_trackfile, possible_packs, power_caps, numEnduranceLaps, end_header)
    autoX_data = optimize_autoX(autoX_trackfile, possible_packs, power_caps, numAutoXLaps, autoX_header)
    accel_data = optimize_accel(possible_packs, numAutoXLaps)



    # Calculate the points

    # Convert all our data to floats
    endurance_times = endurance_data.loc[:, "Endurance Laptime (s)"]
    endurance_times = endurance_times.astype(float)

    endurance_energies = endurance_data.loc[:, "Endurance Energy (kWh)"]
    endurance_energies = endurance_energies.astype(float)

    autoX_times = autoX_data.loc[:, "AutoX Laptime (s)"] 
    autoX_times = autoX_times.astype(float)

    accel_times = accel_data.loc[:, "Accel Laptime (s)"]
    accel_times = accel_times.astype(float)

    end_points = []
    eff_points = []
    autoX_points = []
    total_points = []
    for i in range(len(endurance_times)):
        points = points_estimate(numEnduranceLaps, endurance_times[i], endurance_energies[i], autoX_times[i], accel_times[i])
        end_points.append(points[0])
        autoX_points.append(points[1])
        eff_points.append(points[2])
        total_points.append(points[3])

    points_header = ['Endurance Pts', 'AutoX Pts', 'Efficiency Pts', 'Total Pts']

    points_data = pd.DataFrame(data=zip(end_points, autoX_points, eff_points, total_points), columns=points_header)

    output_df = pd.concat([endurance_data, autoX_data, accel_data, points_data], 
                  axis = 1)


    # Output all the data, inclusding the points
    writer = pd.ExcelWriter('accumulator_points_sims.xlsx', engine='xlsxwriter')
    output_df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()



def skidpad_test():
    high_downforce_reference_file = 'AeroLap.xlsx'
    no_aero_reference_file = 'NoAeroLap.xlsx'

    files_to_compare = [high_downforce_reference_file, no_aero_reference_file]

    laptimes = []
    energies = []

    for filename in files_to_compare:

        data = pd.io.excel.read_excel(filename)

        # Convert all our data to floats
        times = data.loc[:, "TimeStamp"]                                    # time in seconds
        times = times.astype(float)

        # Convert all our data to floats
        total_energy_depleted = data.loc[:, "Total_Energy"]                 # energy in joules
        total_energy_depleted = total_energy_depleted.astype(float)
        total_energy_depleted = total_energy_depleted/(3.6*10**6)           # energy in kWh



        skidpad_laptime = times.iloc[-1] - times.iloc[0]
        skidpad_energy = total_energy_depleted.iloc[-1] - total_energy_depleted.iloc[0]      

        laptimes.append(skidpad_laptime)
        energies.append(skidpad_energy)

    print("With aero: {} s; {} kWh".format(laptimes[0], energies[0]))
    print("Without aero: {} s; {} kWh".format(laptimes[1], energies[1]))


'''

With aero: 5.88 s; 0.0029250976696563887 kWh
Without aero: 6.2 s; 0.0063582105931661095 kWh


We have about a 2x energy discrepancy between the two, which we don't see in the point mass model.

I think it's best to scale our future energy outputs using the "without aero" data, as this lines up the best with info from RIT
-> Info we used to scale our simulated energy output using their car to RIT's reported energy output
-> We only did a few trials of the experiment in person, so we excpect some variation 
-> We'd like to see more tests to hone in on this 

High downforce:
3.8232602907860653 s;    0.015720931645166036 kWh



No aero:
3.901133239998655 s;    0.01552799231063735 kWh


Skidpad scale factor: 

0.0063582105931661095 / 0.01552799231063735

energy_new = energy_predicted * skidpad scale factor



'''

#skidpad_test()

aero_sweep()

#autoX_sweep()

#points_from_spreadsheet()




'''
To validate aero package addition / removal

Use the test we got; compare speed, energy, points with and without endurance
-> Compare all of this with what our sim predicts



Ming's Contact Notes
- We're in a pickle, debating whether to keep or use aero
- Have you tried running a full endurance?
- What are you simming this in?
- The sim is benchmarked with past results of other teams
- We're looking at 24 kW power limit
    "That's really low. You should be fine"
    -> ****I wanna do testing at high power cap and see how the accumulator drains********

- Have you tried running without aero?
- They're at 6.4 kWh
- They have no aero, but in their sims, they were well under margin
- The main limiting factor was temperature 
    -> Check how we can minimize temperature
    -> VTC6s and Emrax 228 are lowkey scary, not optimized. It will lead to more temperature 

    
We're primarily suspicious of our energy consumption. 
    -> ***Check the energy with the skidpad.****
    ->  This is the best testing we can have

    -> Especially with all our sim validation, our angle could be: 
        "the main weakpoint of our sims is the energy drain from the accumulator. We haven't done sufficient testing on this front, 
        but when we compare our actual and simulated energy consumptions for the skidpad, we see simulated consumption is greater"  

"Best bet is just drive slow"

Do you have any strategies to help mitigate browning out as the pack voltage gets low?
    -> They usually don't run into an issue except for the minimum allowed voltage
    -> They have a 300V pack
    -> Their pack doesn't operate at a point where it heats up much; it helps them avoid active cooling 

Faster = burning more energy; slower = aero isn't helping 


Chris's note
    -> They were very confident they'd finish endurance without aero, with a high power cap, with a much heavier car, and a lower pack capacity**
    -> We predict a 5.477946437 kWh consumption, they had a 5.4 kWh pack
'''


'''
Skidpad energy and laptime from sim


'''


'''
We have a < 0.1 s laptime difference between aero and no aero for our simulated skidpad  ; extra downforce doesn't help much
    -> We get laptimes of about 3.8 seconds, significantly faster than the real thing--could be grip factor, but honestly this is a tires issue
            We do not incorporate Pacejka or any of our fancy two-track tire stuff here
    -> We're not grip-limited; we are 
    -> Imma be real Ben, my brain hurts and idk why this discrepancy is so small besides point mass bad
    -> The average velocities of these runs were about 17-19 m/s (no aero), (high downforce); at this speed, drag shouldn't be thaaat significant
    -> 

In reality, we get a ~0.3 s laptime difference between aero and non-aero for the skidpad; a 5% difference 
    -> The actual laptimes Chris got were about 6 s
    -> 

'''