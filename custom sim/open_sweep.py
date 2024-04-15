'''
Rudimentary version of OpenAll for the custom point mass model

'''
import vehicle
import open_loop
import numpy
import pandas as pd

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
    

def main():
    # loop over various motor_curves
    power_caps = [80, 70, 60, 50, 40, 30, 20, 10]

    # loop over various masses
    masses_to_sample = 50
    masses = numpy.linspace(300, 400, masses_to_sample)

    # output vectors of the sim for easier output in excel 
    times = []
    Es = []
    caps = []
    Ms = []

    # for graphing
    energies = numpy.zeros((len(masses), len(power_caps)))
    laptimes = numpy.zeros((len(masses), len(power_caps)))

    for i, new_mass in enumerate(masses):
        for j, power_cap in enumerate(power_caps):
            filename = "motor_curve{}.xlsx".format(power_cap)

            base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\vehicle_files\\"
            filename = base_name + filename

            vehicle.soft_reload(new_mass, filename) # reload our vehicle to have the new data

            laptime, energy = open_loop.simulate()
            laptimes[i, j] = laptime #3D grapher only likes 2D arrays
            energies[i, j] = energy

            times.append(laptime)
            Es.append(energy)
            caps.append(power_cap)
            Ms.append(new_mass)
    
    
    # Output everything using Pandas
    header = ['Mass (kg)', 'Power Cap (kW)', 'Laptime (s)', 'Energy Consumed (kWh)']

    m = numpy.vstack(Ms)
    cap = numpy.vstack(caps)
    t = numpy.vstack(times)
    e = numpy.vstack(Es)

    df0 = numpy.concatenate((m, cap, t, e), axis=1)
    df1 = pd.DataFrame(df0)
    writer = pd.ExcelWriter('Sweep_Data.xlsx', engine='xlsxwriter')

    df1.to_excel(writer, sheet_name='Sheet1', index=False, header=header)
    writer.close()
    
    #plot everything 

    fig = plt.figure()

    Ms_grid, caps_grid = numpy.meshgrid(masses, power_caps)
    
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    


    # plotting
    surf = ax.plot_surface(Ms_grid, caps_grid, laptimes.T, cmap='cool', alpha=0.8, 
                           facecolors=plt.cm.cool(energies.T / numpy.max(energies))) #add energy as a color metric
    # colorbar set up
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='cool', norm=plt.Normalize(numpy.min(energies), numpy.max(energies))), ax=ax)
    cbar.set_label('Energy (kWh)')

    ax.set_title('Laptimes and Energies')
    ax.set_xlabel('Masses (kg)', fontsize=12)
    ax.set_ylabel('Power Caps (kW)', fontsize=12)
    ax.set_zlabel('Laptimes (s)', fontsize=12)

    plt.show()
    

main()