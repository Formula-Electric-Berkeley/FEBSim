from typing import List
import sys, os

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, least_squares
from sklearn.metrics import r2_score
import plotly.graph_objects as go # Nick did this, idk if it's standard but it's cute cause then you get go.Figure()
from plotly.subplots import make_subplots
import plotly

# plotly.io.orca.config.executable = '../../../../AppData/Local/Programs/orca'
# C:\Users\qhcbe\Documents\coding\FEB\FEBSim\TireSim
def format_float(x, sig_digits=3):
    # Round to 3 significant digits
    rounded = f"{x:5G}"
    return str(float(rounded))

# Define the Pacejka Magic Formula for lateral force (FY) with five coefficients
def pacejka_model(alpha, B, C, D, E, F):
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))) + F)

def pacejka_derivative(alpha, B, C, D, E, F):
    return -(C * D * (E * (B - B / ((B ** 2) * (alpha ** 2) + 1)) - B) * np.cos(C * np.arctan(E * (B * alpha - np.arctan(B * alpha)) - B * alpha) - F)) / (((E * (B * alpha - np.arctan(B * alpha)) - B * alpha) ** 2) + 1)

def quad_fit(Fz, a0, a1, a2):
    return a0 + a1 * Fz + a2 * (Fz ** 2)

def lin_fit(Fz, a0, a1):
    return a0 + Fz * a1

def exp_fit(Fz, a0, a1, a2):
    return a0 + a1 * (np.e ** (a2 * Fz))

def fit_print(ftype, *params):
    params = np.vectorize(format_float)(params)
    if ftype == quad_fit:
        return f"{params[0]} + {params[1]} F_z + {params[2]} F_z^2"
    elif ftype == lin_fit:
        return f"{params[0]} + {params[1]} F_z"
    else:
        return f"{params[0]} + {params[1]} exp({params[2]} F_z)"

np.set_printoptions(suppress=True, precision=5) # Disable scientific notation

FZ_BIN_COUNT = 5
BIN_COLORS = ["red", "green", "blue", "purple", "pink", "orange", "black", "gray"]
GRAPH_COLS = 3
GRAPH_ROWS = 2

PACEJKA_PARAMS_NAMES = ["B", "C", "D", "E", "F"]
PACEJKA_PARAMS_GUESS = [-0.1, 0.1, 2000, 0.3, 0]  # Update as needed

PACEJKA_PARAM_FIT_FNS = [quad_fit, quad_fit, lin_fit, exp_fit, quad_fit]
PACEJKA_PARAM_FIT_GUESS = [
    [-0.37, -0.00045, -0.0000002],
    [0, 0, 0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0]
]

# RUN_NUM = 9 # First nine runs are all using our desired 16 x 7.5 tires
SELECTED_RUNS = [4, 5, 6, 8, 9]

def get_run_data(selected_runs : List[int]) -> pd.DataFrame:
    metric_data = pd.DataFrame()

    for rn in selected_runs:
        # Import the data
        metric_datum = pd.read_csv(f'./RunData_Cornering_ASCII_SI_Round9/B2356run{rn}.dat', skiprows=1, sep='\t')

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
        Q1 = metric_datum[['SA deg', 'FY N']].quantile(0.25)
        Q3 = metric_datum[['SA deg', 'FY N']].quantile(0.75)
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
    metric_data_binned = np.zeros(len(Fz_bins), dtype=pd.DataFrame)
    # Sort data into bins, and find coefficients
    for i in range(len(Fz_bins)):
        Fz_bin = Fz_bins[i]

        lb = Fz_bin[0]
        ub = Fz_bin[1]

        data_binned = metric_data[(lb <= metric_data["FZ N"]) & (metric_data["FZ N"] < ub)][["SA deg", "FY N", "FZ N"]]

        metric_data_binned[i] = data_binned
    
    return metric_data_binned

def get_pacejka_params(metric_data : pd.DataFrame) -> tuple[float, float, float, float]:
    params_optimal, _ = curve_fit(pacejka_model, metric_data["SA deg"], metric_data["FY N"], p0=PACEJKA_PARAMS_GUESS, maxfev=100000)
    return params_optimal

def single_fig_color_gradient(metric_data : pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=metric_data["SA deg"],
            y=metric_data["FY N"],
            mode="markers", 
            marker=dict(color=metric_data["FZ N"], colorscale="Viridis", showscale=True, colorbar=dict(title="Load Force (N)")),
        )
    )

    return fig

def bin_figs(data_binned : List[pd.DataFrame], params_binned : List[tuple[float, float, float, float ,float]], Fz_bins : List[tuple[float, float]]) -> go.Figure:
    fig = make_subplots(rows=GRAPH_ROWS, cols=GRAPH_COLS, 
                        subplot_titles=[f"Load force in [{lb:.2f}, {ub:.2f})" for lb, ub in Fz_bins]
    )

    for i in range(FZ_BIN_COUNT):
        data_bin = data_binned[i]
        graph_domain = np.linspace(min(data_bin["SA deg"]) - 10, max(data_bin["SA deg"]) + 10, 500)

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

    fig.update_xaxes(title_text="Slip Angle (deg)")
    fig.update_yaxes(title_text="Lateral Force (N)")
    fig.update_layout(title_text=f"Lateral Force over Slip Angle for Load Force bins")
    
    return fig

def bin_derivative_figs(data_binned : List[pd.DataFrame], params_binned : List[tuple[float, float, float, float ,float]], Fz_bins : List[tuple[float, float]]) -> go.Figure:
    fig = make_subplots(rows=GRAPH_ROWS, cols=GRAPH_COLS, 
                        subplot_titles=[f"Load force in [{lb:.2f}, {ub:.2f})" for lb, ub in Fz_bins]
    )

    for i in range(FZ_BIN_COUNT):
        data_bin = data_binned[i]
        graph_domain = np.linspace(min(data_bin["SA deg"]) - 10, max(data_bin["SA deg"]) + 10, 500)

        # Draw pacejka derivative
        fig.add_trace(
            go.Scatter(
                x=graph_domain, 
                y=pacejka_derivative(graph_domain, *params_binned[i]), 
                mode="lines", 
                marker=dict(color="yellow"), 
                name=f'Derivative for bin {i}'
            ), 
            row=i // GRAPH_COLS + 1, 
            col=i % GRAPH_COLS + 1
        )

    fig.update_xaxes(title_text="Slip Angle (deg)")
    fig.update_yaxes(title_text="d(Fy)/d(SA) (N / deg)")
    fig.update_layout(title_text=f"Instantaneous Cornering Stiffness")
    
    return fig

def bins_single_fig(data_binned : List[pd.DataFrame], params_binned : List[tuple[float, float, float, float ,float]], Fz_bins : List[tuple[float, float]]) -> go.Figure:
    fig = go.Figure()
    graph_domain = np.linspace(min(min(data_bin["SA deg"]) for data_bin in data_binned), max(max(data_bin["SA deg"]) for data_bin in data_binned), 500)

    colors = [
        ("brown", "salmon"),
        ("midnightblue", "royalblue"),
        ("seagreen", "olive"),
        ("purple", "violet"),
        ("black", "gray")
    ]

    for i in range(FZ_BIN_COUNT):
        data_bin = data_binned[i]

        # Draw scatterplot of data
        # fig.add_trace(
        #     go.Scatter( x=data_bin["SA deg"], 
        #                 y=data_bin["FY N"], 
        #                 mode="markers", 
        #                 marker=dict(color=colors[i][0]), 
        #                 name=f'Data, Load force = {(Fz_bins[i][0] + Fz_bins[i][1]) // 2}'
        #     )
        # )

        # Draw pacejka curve
        fig.add_trace(
            go.Scatter(
                x=graph_domain, 
                y=pacejka_model(graph_domain, *params_binned[i]), 
                mode="lines", 
                marker=dict(
                    color=BIN_COLORS[i]
                    ), 
                name=f'{(Fz_bins[i][0] + Fz_bins[i][1]) // 2}',
                legendrank=100  # try a high value to prioritize
            )
        )

    fig.update_xaxes(title_text="Slip Angle (deg)")
    fig.update_yaxes(title_text="Lateral Force (N)")
    fig.update_layout(
        title_text=f"Lateral Force vs Slip Angle",
        legend=dict(x=0.93,y=0.96),
        legend_title_text='Fz (N)',
    )
    fig.update_traces(line={'width': 3})
    
    return fig

def coefficient_figs(params_binned : List[tuple[float, float, float, float, float]], Fz_bins : List[tuple[float, float]]) -> go.Figure:
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{PACEJKA_PARAMS_NAMES[i]} value over load force bins" for i in range(5)])

    for i in range(5):
        values_for_this_param = params_binned[:,i] # ith column
        Fz_medians = [(ub + lb) / 2 for lb, ub in Fz_bins]

        fig.add_trace(go.Scatter(x=Fz_medians, y=values_for_this_param), row = i // 3 + 1, col = i % 3 + 1)
        fig.update_xaxes(title_text=f"Fz Bin Median", row = i // 3 + 1, col = i % 3 + 1)
        fig.update_yaxes(title_text=f"Value of param {PACEJKA_PARAMS_NAMES[i]}", row = i // 3 + 1, col = i % 3 + 1)

        coefficients, _ = curve_fit(PACEJKA_PARAM_FIT_FNS[i], Fz_medians, values_for_this_param, maxfev=150000, p0=PACEJKA_PARAM_FIT_GUESS[i])
        Fz_domain = np.linspace(Fz_medians[0], Fz_medians[-1], 50)
        fig.add_trace(go.Scatter(x=Fz_domain, y=PACEJKA_PARAM_FIT_FNS[i](Fz_domain, *coefficients)), row = i // 3 + 1, col = i % 3 + 1)

        print(f"Pacejka params for {PACEJKA_PARAMS_NAMES[i]}:", coefficients)
        print(fit_print(PACEJKA_PARAM_FIT_FNS[i], *coefficients))

    fig.update_layout(title_text=f"Values for Pacejka params over Fz {SELECTED_RUNS}")

    return fig

def find_radius_conversion():
    starting_radius = 18 # in
    target_radius = 16 # in

    starting_radius_data = get_run_data([68, 71, 72, 73])
    target_radius_data = get_run_data([4, 5, 6, 8, 9])

def surface_fig(all_data : pd.DataFrame, params_binned : List[tuple[float, float, float, float ,float]], Fz_bins : List[tuple[float, float]]) -> go.Figure:
    hyper_coefficients = []
    
    for i in range(5):
        values_for_this_param = params_binned[:,i] # ith column
        Fz_medians = [(ub + lb) / 2 for lb, ub in Fz_bins]

        coefficients, _ = curve_fit(PACEJKA_PARAM_FIT_FNS[i], Fz_medians, values_for_this_param, maxfev=150000, p0=PACEJKA_PARAM_FIT_GUESS[i])
        hyper_coefficients.append(coefficients)

    # Define the Pacejka Magic Formula for lateral force (FY) with five coefficients
    def mvar_pacejka(alpha, Fz):
        B, C, D, E, F = [PACEJKA_PARAM_FIT_FNS[i](Fz, *hyper_coefficients[i]) for i in range(5)]

        return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))) + F)
    
    def mvar_pacejka_derivative(alpha, Fz):
        B, C, D, E, F = [PACEJKA_PARAM_FIT_FNS[i](Fz, *hyper_coefficients[i]) for i in range(5)]

        return -(C * D * (E * (B - B / ((B ** 2) * (alpha ** 2) + 1)) - B) * np.cos(C * np.arctan(E * (B * alpha - np.arctan(B * alpha)) - B * alpha) - F)) / (((E * (B * alpha - np.arctan(B * alpha)) - B * alpha) ** 2) + 1)


    Fz_domain = np.linspace(all_data["FZ N"].min(), -225, 100)
    SA_domain = np.linspace(all_data["SA deg"].min(), all_data["SA deg"].max(), 100)

    Fy_surface = np.zeros(shape=(len(Fz_domain), len(SA_domain)))
    for i in range(len(Fz_domain)):
        for j in range(len(SA_domain)):
            Fy_surface[j][i] = mvar_pacejka_derivative(SA_domain[j], Fz_domain[i])

    fig = go.Figure(data=[go.Surface(
        x=Fz_domain,
        y=SA_domain,
        z=Fy_surface
    )])

    CA_YELLOW = "#FDB515"
    BK_BLUE = "#002676"

    LABEL_COLOR = BK_BLUE
    TICK_COLOR = CA_YELLOW
    
    SCENE_BACKGROUND = "white" #"Greenscreen", to photoshop out background
    PLANE_COLOR = "#242424"

    fig.update_layout(
        scene=dict(
            bgcolor=SCENE_BACKGROUND,
            xaxis=dict(
                gridcolor=TICK_COLOR,
                title=dict(text='Load Force (N)', font=dict(color=LABEL_COLOR, size=30)),
                tickfont=dict(color=LABEL_COLOR),
                showbackground=True,
                backgroundcolor=PLANE_COLOR
            ),
            yaxis=dict(
                gridcolor=TICK_COLOR,
                title=dict(text='Slip Angle (deg)', font=dict(color=LABEL_COLOR, size=30)),
                tickfont=dict(color=LABEL_COLOR),
                showbackground=True,
                backgroundcolor=PLANE_COLOR
            ),
            zaxis=dict(
                gridcolor=TICK_COLOR,
                title=dict(text='d(Fy)/d(SA) (N/deg)', font=dict(color=LABEL_COLOR, size=30)),
                tickfont=dict(color=LABEL_COLOR),
                showbackground=True,
                backgroundcolor=PLANE_COLOR
            ),
        ),
        title=dict(
            x=0.5,
            xanchor='center',
            text='Instantaneous Cornering Stiffness',
            font=dict(
                size=36,
            )
        )
    )
    
    return fig

def fit_print(ftype, *params):
    params = np.vectorize(format_float)(params)
    if ftype == quad_fit:
        return f"{params[0]} + {params[1]} F_z + {params[2]} F_z^2"
    elif ftype == lin_fit:
        return f"{params[0]} + {params[1]} F_z"
    else:
        return f"{params[0]} + {params[1]} exp({params[2]} F_z)"

# Get all data, filtering for specific outliers.
all_data = get_run_data(SELECTED_RUNS)
all_data = all_data[all_data["FZ N"] < -150]

Fz_bins = [(-1300, -1000), (-1000, -750), (-750, -500), (-500, -300), (-300, -150)] # Empirically found by looking at graph.
all_data_binned = Fz_bin_data(all_data, Fz_bins)

# Find pacejka parameters for each of the binned datasets
params_binned = np.array(list(map(get_pacejka_params, all_data_binned)))
for i in range(FZ_BIN_COUNT):
    print(f"Pacejka params for bin {i}:", params_binned[i])

    r2_val = r2_score(all_data_binned[i]["FY N"], pacejka_model(all_data_binned[i]["SA deg"], *params_binned[i]))
    print("R2:", r2_val)

for i in range(5):
    print(PACEJKA_PARAMS_NAMES[i] + "(F_z) = " + fit_print(PACEJKA_PARAM_FIT_FNS[i], *params_binned[i]))

# Create visualizations

# new_fig = surface_fig(all_data, params_binned, Fz_bins)
# new_fig.write_html("fy_out/figures/derivative_surface.html")

# Define the Pacejka Magic Formula for lateral force (FY) with five coefficients
def mod_pacejka_model(alpha, Fz, B, C, E, F, Dm, Db):
    return (Dm * Fz + Db) * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))) + F)

# Residuals function for least_squares
def residuals(params):
    B, C, E, F = params[:4]  # Global params
    Dm, Db = params[4:6]

    alpha = np.deg2rad(all_data['SA deg'].values)
    Fz = all_data["FZ N"].values
    FY = all_data['FY N'].values
    FY_pred = mod_pacejka_model(alpha, Fz, B, C, E, F, Dm, Db)
    
    return FY_pred - FY

# Initial guess: [B0, C0, E0, F0, D0_dataset0, D0_dataset1, ..., D0_datasetN]
# p0 = [-0.1529849279610816, 1.6566112648010622, 0.35726413344630986, 0.028945686691346905] + [-1.70499, 165.3188]

# # Least squares fitting
# result = least_squares(residuals, p0)

# # Extract fitted parameters
# popt = [float(x) for x in list(result.x)]

# # params_binned = [(popt[0:4] + [popt[i]]) for i in range(4, 4 + len(all_data_binned))]

# print(popt)

# r2_val = r2_score(all_data["FY N"], mod_pacejka_model(all_data["SA deg"], all_data["FZ N"], *popt))
# print("R2:", r2_val)

# new_fig = coefficient_figs(params_binned, Fz_bins)
# new_fig.write_html("fy_out/figures/coefficient_plot.html")

# new_fig = bin_figs(all_data_binned, params_binned, Fz_bins)
# new_fig.write_html("fy_out/figures/binned_data_plot.html")

# bin_derivative_fig = bin_derivative_figs(all_data_binned, params_binned, Fz_bins)
# bin_derivative_fig.write_html("fy_out/figures/bin_derivative_plot.html")

sys.exit(0)