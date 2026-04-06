'''
Generate lap times and related statistics based on various vehicle mass values. Creates plots of statistics over mass and point distribution tables.
'''

import open_loop
import vehicle
import accumulator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Main Function, generates mass vs. laptime, energy drain graphs, generates table of values at each track data collection point
def generate_files(track_type="autocross"):
    if track_type == "autocross":
        laptimes, energy_drain, other_data = sweep_over_masses("autocross")

        # Build the point data dataframe
        points_df = pd.concat(other_data, keys=default)
        from IPython.display import display; display(points_df)
        points_df.to_csv('mass_sweeps_files/autocross--point_data.csv')

        # Display laptime and energy drain graphs
        m, b = np.polyfit(default, laptimes, 1)

        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20, 8))

        ax[0].scatter(default, laptimes, color='blue', label="Points")
        ax[0].plot(default, np.array(default) * m + b, color='red', linestyle='--', label="Least Squares")
        ax[0].set(title="Laptime vs. Mass",
                    xlabel="Mass (kg)",
                    ylabel="Lap Time (s)")
        ax[0].legend()

        m, b = np.polyfit(default, energy_drain, 1)

        ax[1].scatter(default, energy_drain, color='blue', label="Points")
        ax[1].plot(default, np.array(default) * m + b, color='red', linestyle='--', label="Least Squares")
        ax[1].set(title="Energy Drain vs. Mass",
                    xlabel="Mass (kg)",
                    ylabel="Energy Drain (kW)")
        ax[1].legend()

        plt.savefig('mass_sweeps_files/autocross--laptime_energy_plots.png')
        plt.show()
    elif track_type == "endurance":
        laptimes, energy_drain, other_data = sweep_over_masses("endurance")

        # Build the point data dataframe
        points_df = pd.concat(other_data, keys=default)
        from IPython.display import display; display(points_df)
        points_df.to_csv('mass_sweeps_files/autocross--point_data.csv')

        # Display laptime and energy drain graphs
        m, b = np.polyfit(default, laptimes, 1)

        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20, 8))

        ax[0].scatter(default, laptimes, color='blue', label="Points")
        ax[0].plot(default, np.array(default) * m + b, color='red', linestyle='--', label="Least Squares")
        ax[0].set(title="Laptime vs. Mass",
                    xlabel="Mass (kg)",
                    ylabel="Lap Time (s)")
        ax[0].legend()

        m, b = np.polyfit(default, energy_drain, 1)

        ax[1].scatter(default, energy_drain, color='blue', label="Points")
        ax[1].plot(default, np.array(default) * m + b, color='red', linestyle='--', label="Least Squares")
        ax[1].set(title="Energy Drain vs. Mass",
                    xlabel="Mass (kg)",
                    ylabel="Energy Drain (kW)")
        ax[1].legend()

        plt.savefig('mass_sweeps_files/autocross--laptime_energy_plots.png')
        plt.show()
    else:
        raise ValueError("Incorrect Track Type. Please choose \'Autocross\' or \'Endurance\'.") 

# Helper Functions

def sweep_over_masses(track_type, masses=[290 + 5 * i for i in range(7)], pack=accumulator.Pack(accumulator.Molicel_Cell_21700).pack(14, 3, 10)):
    assert track_type == "Autocross" or track_type == "Endurance", "Incorrect Track Type. Please choose \'Autocross\' or \'Endurance\'."
    
    laptimes = []
    energy_drain = []
    other_data = []

    for mass in default:
        remass(mass)
        if track_type == "autocross":
            l, e, o = open_loop.simulate(pack)
        elif track_type == "endurance":
            l, e, o = open_loop.simulate_laps(pack)
        laptimes.append(l)
        energy_drain.append(e)
        other_data.append(o)
    
    return laptimes, energy_drain, other_data

def remass(mass):
    vehicle.soft_reload(mass, 80)