from typing import List
import sys

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import plotly.graph_objects as go # Nick did this, idk if it's standard but it's cute cause then you get go.Figure()
from plotly.subplots import make_subplots

# Define the Pacejka Magic Formula for lateral force (FY) with five coefficients
def pacejka_model(sr, B, C, D, E, F):
    return D * np.sin(C * np.arctan(B * sr - E * (B * sr - np.arctan(B * sr))) + F)

def B_fit(Fz, B1, B2):
    return B1 + Fz * B2

def C_fit(Fz, C1, C2):
    return C1 + Fz * C2

def D_fit(Fz, D1, D2):
    return (D1 * Fz) + D2 * (Fz ** 2)

def E_fit(Fz, E1, E2):
    return E1 + Fz * E2

np.set_printoptions(suppress=True) # Disable scientific notation

FZ_BIN_COUNT = 5
BIN_COLORS = ["red", "green", "blue", "purple", "pink", "orange", "black", "gray"]
GRAPH_COLS = 3
GRAPH_ROWS = 2

PACEJKA_PARAMS_NAMES = ["B", "C", "D", "E", "F"]
PACEJKA_PARAMS_GUESS = [-0.1, 0.1, 2000, 0.3, 0]  # Update as needed
PACEJKA_PARAM_FIT_FNS = [B_fit, C_fit, D_fit, E_fit]

# RUN_NUM = 9
# SELECTED_RUNS = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]
 
SELECTED_RUNS = [68, 71, 72, 73]

def get_run_data(selected_runs : List[int]) -> pd.DataFrame:
    metric_data = pd.DataFrame()

    for rn in selected_runs:
        # Import the data
        metric_datum = pd.read_csv(f'./RawData_DriveBrake_ASCII_SI_Round9/B2356raw{rn}.dat', skiprows=1, sep='\t')

        # Rename columns to have value and units, and remove the previous row for units
        metric_datum.rename(columns={col: f"{col} {metric_datum[col][0]}" for col in metric_datum.columns}, inplace=True)
        metric_datum = metric_datum.drop(0)

        # Convert all values to floats
        metric_datum = metric_datum.apply(pd.to_numeric, errors='coerce')

        # Drop any rows with NaN values (if necessary)
        metric_datum = metric_datum.dropna()

        # Reset index
        metric_datum.reset_index(drop=True, inplace=True)

        # print(metric_datum.columns)
        # print(metric_datum)

        # IQR filter Slip Angle and Fy
        # print(metric_datum.columns)

        Q1 = metric_datum[['SR SAE', 'FX N']].quantile(0.25)
        Q3 = metric_datum[['SR SAE', 'FX N']].quantile(0.75)
        IQR = Q3 - Q1

        # metric_datum = metric_datum[~((metric_datum[['SA deg', 'FY N']] < (Q1 - 1.5 * IQR)) | (metric_datum[['SA deg', 'FY N']] > (Q3 + 1.5 * IQR))).any(axis=1)]

        metric_data = pd.concat([metric_data, metric_datum])
        print(len(metric_datum))

    return metric_data

def get_bins(metric_data : pd.DataFrame) -> List[tuple[int, int]]:
    # Define bins by evenly splitting range

    # Fz_min = metric_data["FZ N"].min()
    # Fz_max = metric_data["FZ N"].max()
    # Fz_range = Fz_max - Fz_min

    # Fz_bin_points = np.linspace(Fz_min, Fz_max, FZ_BIN_COUNT)

    # Define bins by splitting by percentages of data in each bin

    Fz_bin_points = [metric_data["FZ N"].quantile(p) for p in np.linspace(0, 1, FZ_BIN_COUNT + 1)]

    Fz_bins = [[lb, ub] for lb, ub in zip(Fz_bin_points[:-1], Fz_bin_points[1:])]

    return Fz_bins

def Fz_bin_data(metric_data : pd.DataFrame, Fz_bins : List[List[int]]) -> List[pd.DataFrame]:
    metric_data_binned = []
    # Sort data into bins, and find coefficients
    for Fz_bin in Fz_bins:
        lb = Fz_bin[0]
        ub = Fz_bin[1]

        data_binned = metric_data[(lb <= metric_data["FZ N"]) & (metric_data["FZ N"] < ub)][["SA deg", "FY N", "FZ N"]]

        metric_data_binned.append(data_binned)
    
    return metric_data_binned

def get_pacejka_params(metric_data : pd.DataFrame) -> tuple[float, float, float, float]:
    params_optimal, _ = curve_fit(pacejka_model, metric_data["SA deg"], metric_data["FY N"], p0=PACEJKA_PARAMS_GUESS, maxfev=100000)
    return params_optimal

def single_fig_color_gradient(metric_data : pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=metric_data["SR SAE"],
            y=metric_data["FX N"],
            mode="markers", 
            marker=dict(color=metric_data["FZ N"], colorscale="Viridis", showscale=True, colorbar=dict(title="Load Force (N)")),
        )
    )

    return fig

def bin_figs(data_binned : List[pd.DataFrame], params_binned : List[tuple[float, float, float, float ,float]], Fz_bins : List[tuple[float, float]]) -> go.Figure:
    fig = make_subplots(rows=GRAPH_ROWS, cols=GRAPH_COLS, subplot_titles=[f"Fy over SA for Fz in bin [{lb:.2f}, {ub:.2f})" for lb, ub in Fz_bins])

    for i in range(FZ_BIN_COUNT):
        data_bin = data_binned[i]
        graph_domain = np.linspace(min(data_bin["SA deg"]), max(data_bin["SA deg"]), 500)

        # Draw scatterplot of data
        fig.add_trace(
            go.Scatter( x=data_bin["SA deg"], 
                        y=data_bin["FY N"], 
                        mode="markers", 
                        # marker=dict(
                        #     color=data_bin["FZ N"], 
                        #     colorscale="Viridis", 
                        #     showscale=True, 
                        #     colorbar=dict(title=f"Bin {i + 1}")
                        # ),
                        name=f'Data for bin {i}'
            ), 
            row=i // GRAPH_COLS + 1, 
            col=i % GRAPH_COLS + 1
        )

        # Draw pacejka curve
        fig.add_trace(
            go.Scatter(
                x=graph_domain, 
                y=pacejka_model(graph_domain, *params_binned[i]), 
                mode="lines", 
                marker=dict(color="yellow"), 
                name=f'Fitted curve for bin {i}'
            ), 
            row=i // GRAPH_COLS + 1, 
            col=i % GRAPH_COLS + 1
        )
    
    return fig

def coefficient_figs(params_binned : List[tuple[float, float, float, float, float]], Fz_bins : List[tuple[float, float]]) -> go.Figure:
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{PACEJKA_PARAMS_NAMES[i]} value over load force bins" for i in range(5)])

    for i in range(5):
        Fz_medians = [(ub + lb) / 2 for lb, ub in Fz_bins]

        fig.add_trace(go.Scatter(x=Fz_medians, y=[params_binned[j][i] for j in range(FZ_BIN_COUNT)]), row = i // 3 + 1, col = i % 3 + 1)
        fig.update_xaxes(title_text=f"Fz Bin Median", row = i // 3 + 1, col = i % 3 + 1)
        fig.update_yaxes(title_text=f"Value of param {PACEJKA_PARAMS_NAMES[i]}", row = i // 3 + 1, col = i % 3 + 1)

        # Fit param data for the Pacejka model coefficients, we don't have a fit fn for F not sure why
        if i < 4:
            coefficients, _ = curve_fit(PACEJKA_PARAM_FIT_FNS[i], Fz_medians, [params_binned[j][i] for j in range(FZ_BIN_COUNT)])
            Fz_domain = np.linspace(Fz_medians[0], Fz_medians[-1], 50)
            fig.add_trace(go.Scatter(x=Fz_domain, y=PACEJKA_PARAM_FIT_FNS[i](Fz_domain, *coefficients)), row = i // 3 + 1, col = i % 3 + 1)

    fig.update_layout(title_text=f"Binned Fy over SA for Fz bins on runs {SELECTED_RUNS}")

    return fig

all_data = get_run_data(SELECTED_RUNS)
# all_data = all_data[(all_data["FZ N"] > -1000) & (all_data["FZ N"] < -350)]

# fig = single_fig_color_gradient(all_data)

print("MIN:", all_data["FZ N"].min())

fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=all_data[(all_data["FZ N"] > -350) & (all_data["FZ N"] < -0)]["SR SAE"],
#     y=all_data[(all_data["FZ N"] > -350) & (all_data["FZ N"] < -0)]["FX N"],
#     # opacity=0.5,
#     mode="markers",
# ))
# fig.add_trace(go.Scatter(
#     x=all_data[(all_data["FZ N"] > -800) & (all_data["FZ N"] < -350)]["SR SAE"],
#     y=all_data[(all_data["FZ N"] > -800) & (all_data["FZ N"] < -350)]["FX N"],
#     # opacity=0.5,
#     mode="markers",
# ))
fig.add_trace(go.Scatter(
    x=all_data[(all_data["FZ N"] > -1000) & (all_data["FZ N"] < -800)]["SR SAE"],
    y=all_data[(all_data["FZ N"] > -1000) & (all_data["FZ N"] < -800)]["FX N"],
    # opacity=0.5,
    mode="markers", 
))
# fig.add_trace(go.Scatter(
#     x=all_data[(all_data["FZ N"] > -1500) & (all_data["FZ N"] < -1000)]["SR SAE"],
#     y=all_data[(all_data["FZ N"] > -1500) & (all_data["FZ N"] < -1000)]["FX N"],
#     opacity=0.5,
#     mode="markers", 
# ))

fig.write_html("QVis.html")

# all_data = all_data[all_data["FZ N"]]

# Fz_bins = [(-1300, -1000), (-1000, -750), (-750, -500), (-500, -300), (-300, -150)] # Empirically found by looking at graph.
# all_data_binned = Fz_bin_data(all_data, Fz_bins)

# params_binned = [get_pacejka_params(data_bin) for data_bin in all_data_binned]

# for i in range(FZ_BIN_COUNT):
#     print(f"Bin {i}")
#     print("Params:", params_binned[i])

#     r2_val = r2_score(all_data_binned[i]["FY N"], pacejka_model(all_data_binned[i]["SA deg"], *params_binned[i]))
#     print("R2:", r2_val)


# new_fig = coefficient_figs(params_binned, Fz_bins)
# new_fig.write_html("QVis.html")

sys.exit(0)