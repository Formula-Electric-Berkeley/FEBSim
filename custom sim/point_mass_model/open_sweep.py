'''
Rudimentary version of OpenAll for the custom point mass model

'''
import vehicle as vehicle
import open_loop
import numpy
import pandas as pd
import track

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
def points_estimate(reference_file, numLaps, time_endurance, time_autoX, time_acceleration, energy_endurance): 
    # Track excel file selection
    
    #info = read_info(reference_file,'Sheet1')

    #values = info.loc[:, "Value"]

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
    tEnd = time_endurance*numLaps
    #eEnd = energy_endurance*numLaps # already taken care of

    # Calculate points for Endurance, Autocross, and Acceleration (from rulebook)
    ptsEnd = min(250, 250*((minimum_endurance_time*1.45/tEnd)-1)/((minimum_endurance_time*1.45/minimum_endurance_time)-1))
    ptsAutoX = min(125, 118.5*((minimum_autoX_time*1.45/time_autoX)-1)/((minimum_autoX_time*1.45/minimum_autoX_time)-1) + 6.5)
    ptsAcc = min(100, 95.5*((minimum_acceleration_time*1.5/time_acceleration)-1)/((minimum_acceleration_time*1.5/minimum_acceleration_time)-1) + 4.5)

    # Calculate efficiency factor and points
    effFactor = (minimum_endurance_time/tEnd)*(minimum_energy_endurance/energy_endurance)
    ptsEff = 0.0 #min(100, 100*(effFactor)/(efficiency_factor_maximum))

    
    # Return calculated points
    pts = [ptsEnd, ptsAutoX, ptsEff, ptsAcc, 0]

    # Dynamic points:
    dynamic_pts = numpy.sum(pts[:3])

    ptsTot = numpy.sum(pts)

    return (ptsEnd, ptsAutoX, ptsTot)


def main():
    # used for points estimation
    ptsRef_filename = 'SN3_Points_Reference.xlsx'
    numLaps = 22

    # loop over various motor_curves
    power_caps = [80, 70, 60, 50, 40, 30, 20, 10]

    # loop over various masses
    masses_to_sample = 50
    masses = numpy.linspace(200, 300, masses_to_sample)

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
    end_energies = numpy.zeros((len(masses), len(power_caps)))
    end_laptimes = numpy.zeros((len(masses), len(power_caps)))

    autoX_energies = numpy.zeros((len(masses), len(power_caps)))
    autoX_laptimes = numpy.zeros((len(masses), len(power_caps)))
    
    accel_laptimes = numpy.zeros((len(masses), len(power_caps)))

    end_points = numpy.zeros((len(masses), len(power_caps)))
    autoX_points = numpy.zeros((len(masses), len(power_caps)))
    tot_points = numpy.zeros((len(masses), len(power_caps)))

    #start endurance
    track.reload('Michigan 2014.xlsx')

    for i, new_mass in enumerate(masses):
        for j, power_cap in enumerate(power_caps):
            filename = "motor_curve{}.xlsx".format(power_cap)

            base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\vehicle_files\\"
            filename = base_name + filename

            vehicle.soft_reload(new_mass, filename) # reload our vehicle to have the new data

            laptime, energy = open_loop.simulate()
            end_laptimes[i, j] = laptime #3D grapher only likes 2D arrays
            end_energies[i, j] = energy

            end_times.append(laptime)
            end_Es.append(energy)
            caps.append(power_cap)
            Ms.append(new_mass)

    
    # start AutoX
    track.reload('Michigan_2022_AutoX.xlsx')
    for i, new_mass in enumerate(masses):
            for j, power_cap in enumerate(power_caps):
                filename = "motor_curve{}.xlsx".format(power_cap)

                base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\vehicle_files\\"
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
                points = points_estimate(ptsRef_filename, numLaps, end_laptimes[i, j], laptime, accel_time, end_energies[i, j]) #no accel for now
                end_points[i, j] = points[0]
                autoX_points[i, j] = points[1]
                tot_points[i, j] = points[2]
                p1.append(points[0])
                p2.append(points[1])
                p3.append(points[2])


        
    # Output everything using Pandas
    header = ['Mass (kg)', 'Power Cap (kW)', 'Endurance Laptime (s)', 'Endurance Energy (kWh)', 'Endurance Points', 
              'AutoX Laptime (s)', 'AutoX Energy (kWh)', 'AutoX Points', 'Accel Time', 'Total Points']

    m = numpy.vstack(Ms)
    cap = numpy.vstack(caps)
    t1 = numpy.vstack(end_times)
    e1 = numpy.vstack(end_Es)
    t2 = numpy.vstack(autoX_times)
    e2 = numpy.vstack(autoX_Es)
    t3 = numpy.vstack(accel_times)
    p1 = numpy.vstack(p1)
    p2 = numpy.vstack(p2)
    p3 = numpy.vstack(p3)


    df0 = numpy.concatenate((m, cap, t1, e1, p1, t2, e2, p2, t3, p3), axis=1)
    df1 = pd.DataFrame(df0)
    writer = pd.ExcelWriter('Sweep_Data_5.xlsx', engine='xlsxwriter')

    df1.to_excel(writer, sheet_name='Sheet1', index=False, header=header)
    writer.close()
    
    #plot everything 

    fig = plt.figure()

    Ms_grid, caps_grid = numpy.meshgrid(masses, power_caps)
    
    # syntax for 3-D projection
    ax1 = fig.add_subplot(231, projection ='3d')
    ax2 = fig.add_subplot(232, projection ='3d')
    ax3 = fig.add_subplot(233, projection ='3d')
    ax4 = fig.add_subplot(234, projection ='3d')
    ax5 = fig.add_subplot(235, projection ='3d')
    ax6 = fig.add_subplot(236, projection ='3d')


    # plotting
    surf = ax1.plot_surface(Ms_grid, caps_grid, end_laptimes.T, cmap='cool', alpha=0.8, 
                           facecolors=plt.cm.cool(end_energies.T / numpy.max(end_energies))) #add energy as a color metric
    # colorbar set up
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool', norm=plt.Normalize(numpy.min(end_energies), numpy.max(end_energies))), ax=ax1)
    cbar.set_label('Energy (kWh)')

    ax1.set_title('Endurance Laptimes and Energies')
    ax1.set_xlabel('Masses (kg)', fontsize=12)
    ax1.set_ylabel('Power Caps (kW)', fontsize=12)
    ax1.set_zlabel('Laptimes (s)', fontsize=12)

    surf = ax2.plot_surface(Ms_grid, caps_grid, autoX_laptimes.T, cmap='cool', alpha=0.8, 
                           facecolors=plt.cm.cool(autoX_energies.T / numpy.max(autoX_energies))) #add energy as a color metric
    # colorbar set up
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool', norm=plt.Normalize(numpy.min(autoX_energies), numpy.max(autoX_energies))), ax=ax2)
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
    

main()