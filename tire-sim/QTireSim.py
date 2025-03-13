from typing import List

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import plotly.graph_objects as go # Nick did this, idk if it's standard but it's cute cause then you get go.Figure()
from plotly.subplots import make_subplots

# Define the Pacejka Magic Formula for lateral force (FY) with five coefficients
def pacejka_model(alpha, B, C, D, E, F):
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))) + F)

np.set_printoptions(suppress=True) # Disable scientific notation

FZ_BIN_COUNT = 4
BIN_COLORS = ["red", "green", "blue", "purple", "pink", "orange"]

PACEJKA_PARAMS_GUESS = [0.1, 0.1, 100, 0.1, 0]  # Update as needed
PACEJKA_PARAMS_BOUNDS = ([0, 0, 0, 0, 0],  [np.inf, np.inf, np.inf, np.inf, np.inf])

RUN_NUM = 51

# Import the data
metric_data = pd.read_csv(f'./RunData_DriveBrake_ASCII_SI_Round9/B2356run{RUN_NUM}.dat', skiprows=1, sep='\t')

# Rename columns to have value and units, and remove the previous row for units
metric_data.rename(columns={col: f"{col} {metric_data[col][0]}" for col in metric_data.columns}, inplace=True)
metric_data = metric_data.drop(0)

# Convert all values to floats
metric_data = metric_data.apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN values (if necessary)
metric_data = metric_data.dropna()

# Reset index
metric_data.reset_index(drop=True, inplace=True)

print(metric_data.columns)
print(metric_data)

# IQR filter Slip Angle and Fy
Q1 = metric_data[['SA deg', 'FY N']].quantile(0.25)
Q3 = metric_data[['SA deg', 'FY N']].quantile(0.75)
IQR = Q3 - Q1

metric_data = metric_data[~((metric_data[['SA deg', 'FY N']] < (Q1 - 1.5 * IQR)) | (metric_data[['SA deg', 'FY N']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Define bins by evenly splitting range

# Fz_min = metric_data["FZ N"].min()
# Fz_max = metric_data["FZ N"].max()
# Fz_range = Fz_max - Fz_min

# Fz_bin_points = np.linspace(Fz_min, Fz_max, FZ_BIN_COUNT)

# Define bins by splitting by percentages of data in each bin

Fz_bin_points = [metric_data["FZ N"].quantile(p) for p in np.linspace(0, 1, FZ_BIN_COUNT + 1)]

Fz_bins = [[lb, ub] for lb, ub in zip(Fz_bin_points[:-1], Fz_bin_points[1:])]

all_data_binned = []
params_binned = []

# Sort data into bins, and find coefficients
for Fz_bin in Fz_bins:
    lb = Fz_bin[0]
    ub = Fz_bin[1]

    data_binned = metric_data[(lb <= metric_data["FZ N"]) & (metric_data["FZ N"] < ub)][["SA deg", "FY N"]].values.T
    all_data_binned.append(data_binned)

    params_optimal, _ = curve_fit(pacejka_model, data_binned[0], data_binned[1], p0=PACEJKA_PARAMS_GUESS, maxfev=1500000)
    params_binned.append(params_optimal)

# Draw the data
# fig = go.Figure()
fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Fy over SA for Fz in bin [{lb:.2f}, {ub:.2f})" for lb, ub in Fz_bins])

for i in range(FZ_BIN_COUNT):
    data_binned = all_data_binned[i]
    graph_domain = np.linspace(min(data_binned[0]), max(data_binned[0]), 500)

    fig.add_trace(go.Scatter(x=data_binned[0], y=data_binned[1], mode="markers", marker=dict(color=BIN_COLORS[i]), name=f'Data for bin {i}'), row=i // 2 + 1, col=i % 2 + 1)
    fig.add_trace(go.Scatter(x=graph_domain, y=pacejka_model(graph_domain, *params_binned[i]), mode="lines", marker=dict(color="yellow"), name=f'Fitted curve for bin {i}'), row=i // 2 + 1, col=i % 2 + 1)

min_r2 = np.inf

for i in range(FZ_BIN_COUNT):
    print(f"Bin {i}")
    
    print("Params:", params_binned[i])

    r2_val = r2_score(all_data_binned[i][1], pacejka_model(all_data_binned[i][0], *params_binned[i]))
    print("R2:", r2_val)

    if r2_val < min_r2:
        min_r2 = r2_val

print(min_r2)

fig.update_layout(title_text=f"Binned Fy over SA for Fz bins on run {RUN_NUM}")
fig.show()