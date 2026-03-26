from typing import List
import sys
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import plotly.graph_objects as go # Nick did this, idk if it's standard but it's cute cause then you get go.Figure()
# don't think it's standard but it's funny
from plotly.subplots import make_subplots

def format_float(x, sig_digits=3):
    # Round to 3 significant digits
    rounded = f"{x:5G}"
    return str(float(rounded))

# Define the Pacejka Magic Formula for lateral force (FY) with five coefficients
def pacejka_model(kappa, B, C, D, E, F):
    return D * np.sin(C * np.arctan(B * kappa - E * (B * kappa - np.arctan(B * kappa))) + F)

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
pd.options.plotting.backend = "plotly"

FZ_BIN_COUNT = 5
BIN_COLORS = ["red", "green", "blue", "purple", "pink", "orange", "black", "gray"]
GRAPH_COLS = 3
GRAPH_ROWS = 2

PACEJKA_PARAMS_NAMES = ["B", "C", "D", "E", "F"]
PACEJKA_PARAMS_GUESS = [-18, -0.01, 100000, -2, 0.00076]  # Update as needed

PACEJKA_PARAMS_FIT_FNS = [exp_fit, quad_fit, quad_fit, lin_fit, lin_fit]
PACEJKA_PARAMS_FIT_GUESS = [
    [-18, -30, 0.01],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0],
    [0, 0]
]

# runs w/ 18.0X6.0 tire data
SELECTED_RUNS = [72, 73]
k_y = 1.059952828992417 # see notion for derivation

def get_run_data(selected_runs : List[int]) -> pd.DataFrame:
    metric_data = pd.DataFrame()

    for rn in selected_runs:
        # Import the data
        metric_datum = pd.read_csv(f'./RunData_DriveBrake_ASCII_SI_Round9/B2356run{rn}.dat', skiprows=1, sep='\t')

        # Rename columns to have value and units, and remove the previous row for units
        metric_datum.rename(columns={col: f"{col} {metric_datum[col][0]}" for col in metric_datum.columns}, inplace=True)
        metric_datum = metric_datum.drop(0)

        # Convert all values to floats
        metric_datum = metric_datum.apply(pd.to_numeric, errors='coerce')

        # Drop any rows with NaN values (if necessary)
        metric_datum = metric_datum.dropna()

        # Reset index
        metric_datum.reset_index(drop=True, inplace=True)

        # Fill in SL data if necessary
        # if "SL none" not in metric_datum.columns:
        #     metric_datum["SL none"] =  (metric_datum["RE cm"] / metric_datum["RL cm"]) * (metric_datum["SR SAE"] + 1) - 1

        metric_data = pd.concat([metric_data, metric_datum])
        print(len(metric_datum))
    
    
    # Rescale forces for tire diam
    metric_data["FX (unscaled) N"] = metric_data["FX N"]
    metric_data["FX N"] = k_y * metric_data["FX N"]

    print(metric_data)
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

        data_binned = metric_data[(lb <= metric_data["FZ N"]) & (metric_data["FZ N"] < ub)][["SL none", "FX N", "FZ N"]]

        metric_data_binned.append(data_binned)
    
    return metric_data_binned

def get_pacejka_params(metric_data : pd.DataFrame) -> tuple[float, float, float, float]:
    params_optimal, _ = curve_fit(pacejka_model, metric_data["SL none"], metric_data["FX N"], p0=PACEJKA_PARAMS_GUESS, maxfev=100000)
    return params_optimal

def single_fig_color_gradient(metric_data : pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=metric_data["SL none"],
            y=metric_data["FX N"],
            mode="markers", 
            #   ,
        )
    )

    return fig

def bin_figs(data_binned : List[pd.DataFrame], params_binned : List[tuple[float, float, float, float ,float]], Fz_bins : List[tuple[float, float]]) -> go.Figure:
    fig = make_subplots(rows=GRAPH_ROWS, cols=GRAPH_COLS, subplot_titles=[f"Fx over SR for Fz in bin [{lb:.2f}, {ub:.2f})" for lb, ub in Fz_bins])

    for i in range(FZ_BIN_COUNT):
        data_bin = data_binned[i]
        graph_domain = np.linspace(min(data_bin["SL none"]), max(data_bin["SL none"]), 500)

        # Draw scatterplot of data
        fig.add_trace(
            go.Scatter( x=data_bin["SL none"], 
                        y=data_bin["FX N"], 
                        mode="markers", 
                        # marker=dict(
                        #     color=data_bin["FZ N"], 
                        #     colorscale="Viridis", 
                        #     showscale=True, 
                        #     colorbar=dict(title=f"Bin {i + 1}")
                        # ),
                        name=f'Data for bin {i}',
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
        values_for_this_param = params_binned[:,i] # ith column
        Fz_medians = [(ub + lb) / 2 for lb, ub in Fz_bins]

        fig.add_trace(go.Scatter(x=Fz_medians, y=values_for_this_param), row = i // 3 + 1, col = i % 3 + 1)
        fig.update_xaxes(title_text=f"Fz Bin Median", row = i // 3 + 1, col = i % 3 + 1)
        fig.update_yaxes(title_text=f"Value of param {PACEJKA_PARAMS_NAMES[i]}", row = i // 3 + 1, col = i % 3 + 1)

        coefficients, _ = curve_fit(PACEJKA_PARAM_FIT_FNS[i], Fz_medians, values_for_this_param, maxfev=15000000, p0=PACEJKA_PARAM_FIT_GUESS[i])
        Fz_domain = np.linspace(Fz_medians[0], Fz_medians[-1], 50)
        fig.add_trace(go.Scatter(x=Fz_domain, y=PACEJKA_PARAM_FIT_FNS[i](Fz_domain, *coefficients)), row = i // 3 + 1, col = i % 3 + 1)

        print(f"Pacejka params for {PACEJKA_PARAMS_NAMES[i]}:", coefficients)
        print(fit_print(PACEJKA_PARAM_FIT_FNS[i], *coefficients))

    fig.update_layout(title_text=f"Values for Pacejka params over Fz {SELECTED_RUNS}")

    return fig

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

    Fz_domain = np.linspace(all_data["FZ N"].min(), -225, 100)
    SA_domain = np.linspace(all_data["SL none"].min(), all_data["SL none"].max(), 100)

    Fy_surface = np.zeros(shape=(len(Fz_domain), len(SA_domain)))
    for i in range(len(Fz_domain)):
        for j in range(len(SA_domain)):
            Fy_surface[j][i] = mvar_pacejka(SA_domain[j], Fz_domain[i])

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
                title=dict(text='Slip Ratio', font=dict(color=LABEL_COLOR, size=30)),
                tickfont=dict(color=LABEL_COLOR),
                showbackground=True,
                backgroundcolor=PLANE_COLOR
            ),
            zaxis=dict(
                gridcolor=TICK_COLOR,
                title=dict(text='Longitudinal Force (N)', font=dict(color=LABEL_COLOR, size=30)),
                tickfont=dict(color=LABEL_COLOR),
                showbackground=True,
                backgroundcolor=PLANE_COLOR
            ),
        ),
        title=dict(
            x=0.5,
            xanchor='center',
            text='Longitudinal Force vs Load Force and Slip Ratio',
            font=dict(
                size=36,
            )
        )
    )
    
    return fig

all_data = get_run_data(SELECTED_RUNS)

special_data = all_data[all_data["SL none"].notnull()]

# desired_bin = (-np.inf, -900)

# all_data = all_data[(all_data["FZ N"] > desired_bin[0]) & (all_data["FZ N"] < desired_bin[1])]
# new_fig = single_fig_color_gradient(all_data)

Fz_bins = [(-2000, -900), (-900, -725), (-725, -400), (-400, -250), (-250, 0)] # Empirically found by looking at graph.
all_data_binned = Fz_bin_data(all_data, Fz_bins)

params_binned = np.array(list(map(get_pacejka_params, all_data_binned)))

timestamp = datetime.now().strftime("%m%d_%H%M%S")

hyper_coefficients = []

for i in range(5):
    values_for_this_param = params_binned[:,i] # ith column
    Fz_medians = [(ub + lb) / 2 for lb, ub in Fz_bins]

    coefficients, _ = curve_fit(PACEJKA_PARAMS_FIT_FNS[i], Fz_medians, values_for_this_param, maxfev=150000, p0=PACEJKA_PARAMS_FIT_GUESS[i])
    hyper_coefficients.append(coefficients)
    
surf_fig = surface_fig(all_data, hyper_coefficients, Fz_bins, 0, -800)

surf_fig.write_html(f"fy_out/figures/fy_sens_surf_({timestamp}).html")

# bin_fig = bin_figs(all_data_binned, params_binned, Fz_bins)
# coeff_fig = coefficient_figs(params_binned, Fz_bins)

# for i in range(FZ_BIN_COUNT):
#     print(f"Bin {i}")
#     print("Params:", params_binned[i])

#     r2_val = r2_score(all_data_binned[i]["FX N"], pacejka_model(all_data_binned[i]["SL none"], *params_binned[i]))
#     print("R2:", r2_val)


# # bin_fig.write_html("fx_out/bins_plot.html")
# # coeff_fig.write_html("fx_out/coeff_plot.html")

# surface = surface_fig(all_data, params_binned, Fz_bins)

# surface.write_html("fx_out/surface_fig.html")

# sys.exit(0)