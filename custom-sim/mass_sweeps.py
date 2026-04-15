'''
Generate lap times and related statistics based on various vehicle mass values. Creates plots of statistics over mass and point distribution tables.
'''

import open_loop
import vehicle
import accumulator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Main Function, generates mass vs. laptime, energy drain graphs, generates table of values at each track data collection point
def simulate_event(track_type="Autocross", masses=[290 + 5 * i for i in range(7)]):
    if track_type in ("Autocross", "Endurance"):
        laptimes, energy_drain, other_data = sweep_over_masses(track_type, masses)
    else:
        raise ValueError("Incorrect Track Type. Please choose \'Autocross\' or \'Endurance\'.") 
    
    # Calculate scores
    scores = []
    if track_type == "Autocross":
        for l in laptimes:
            scores.append(score(l))
    elif track_type == "Endurance":
        for l in laptimes:
            scores.append(score(l, track_type="Endurance"))

    # Build the point data dataframe
    points_df = pd.concat(other_data, keys=masses)
    from IPython.display import display; display(points_df)
    if track_type == "Autocross":
        points_df.to_csv('mass_sweeps_files/autocross--point_data.csv')
    elif track_type == "Endurance":
        points_df.to_csv('mass_sweeps_files/endurance--point_data.csv')

    # Display laptime and energy drain graphs
    m, b = np.polyfit(masses, scores, 1)

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20, 8))

    ax[0].scatter(masses, scores, color='blue', label="Points")
    ax[0].plot(masses, np.array(masses) * m + b, color='red', linestyle='--', label="Least Squares")
    ax[0].set(title="Scores vs. Mass",
                xlabel="Mass (kg)",
                ylabel="Score (pts)")
    ax[0].legend()

    m, b = np.polyfit(masses, energy_drain, 1)

    ax[1].scatter(masses, energy_drain, color='blue', label="Points")
    ax[1].plot(masses, np.array(masses) * m + b, color='red', linestyle='--', label="Least Squares")
    ax[1].set(title="Energy Drain vs. Mass",
                xlabel="Mass (kg)",
                ylabel="Energy Drain (kW)")
    ax[1].legend()

    if track_type == "Autocross":
        plt.savefig('mass_sweeps_files/autocross--laptime_energy_plots.png')
    elif track_type == "Endurance":
        plt.savefig('mass_sweeps_files/endurance--laptime_energy_plots.png')
    plt.show()

# Helper Functions

def sweep_over_masses(track_type, masses, pack=accumulator.Pack(accumulator.Molicel_Cell_21700)):
    assert track_type == "Autocross" or track_type == "Endurance", "Incorrect Track Type. Please choose \'Autocross\' or \'Endurance\'."

    laptimes = []
    energy_drain = []
    other_data = []

    for mass in masses:
        pack.pack(14, 3, 10)
        remass(mass)

        if track_type == "Autocross":
            l, e, o = open_loop.simulate(pack)
        elif track_type == "Endurance":
            l, e, pf, o = open_loop.simulate_laps(pack)
        laptimes.append(l)
        energy_drain.append(e)
        other_data.append(o)
    
    return laptimes, energy_drain, other_data

def remass(mass):
    vehicle.soft_reload(mass, 80)

def score(laptime, doo=0, oc=0, other_penalty=0, laps=22, track_type="Autocross"):
    if track_type == "Autocross":
        TMIN = 45.734
        TMAX = TMIN * 1.45

        tyour = laptime + (doo * 2) + (oc * 20) + other_penalty

        if tyour > TMAX:
            score = 6.5
        else:
            score = 118.5 * ((TMAX / tyour) - 1) / ((TMAX / TMIN) - 1) + 6.5

        return round(score, 2)
    elif track_type == "Endurance":
        TMIN = 1369.936
        TMAX = TMIN * 1.45

        LAPSCORE_SCALE = np.array([[22, 25],
                                   [21, 24],
                                   [20, 23],
                                   [19, 22],
                                   [18, 21],
                                   [11, 14],
                                   [2, 3],
                                   [1, 2]][::-1])
        lapscore_func = CubicSpline(LAPSCORE_SCALE[:, 0], LAPSCORE_SCALE[:, 1])

        if laps == 22:
            tyour = laptime + (doo * 2) + (oc * 20)

            if tyour > TMAX:
                score = 0
            else:
                score = 250 * ((TMAX / tyour) - 1) / ((TMAX / TMIN) - 1)
        else:
            score = 0
        
        score += lapscore_func(laps)

        return round(score, 1)
    else:
        raise ValueError("Incorrect Track Type. Please choose \'Autocross\' or \'Endurance\'.") 