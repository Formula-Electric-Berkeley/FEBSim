from typing import List
import sys

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


#NOTE: CREATE A FOLDER NAMED "htmlGraphs" IN THE tire-sim DIRECTORY TO SAVE THE OUTPUT HTML FILES


np.set_printoptions(suppress=True, precision=5)
pd.options.plotting.backend = "plotly"

GRAPH_COLS = 3
GRAPH_ROWS = 2
IA_BIN_COUNT = 5

# runs 1-9 are w/ 18.0X6.0 tire data
#SELECTED_RUNS = [8]

#Pacjeka Model 
def pacejka_lateral(alpha_deg, B, C, D, E, F):
    """
    5-parameter Magic Formula for lateral force vs slip angle.
    alpha_deg is in degrees (matches your data), and we keep the same units in the model.
    """
    a = alpha_deg
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))) + F)

# Load data
def get_run_data() -> pd.DataFrame:
    """
    Loads data from the specific file B2356run9.dat located in
    ./RunData_Cornering_ASCII_SI_Round9/.
    """
    file_path = 'B2356run9.dat'
    df = pd.read_csv(file_path, skiprows=1, sep='\t', low_memory=False)

    # append unit row into column names, then drop it
    df.rename(columns={col: f"{col} {df[col][0]}" for col in df.columns}, inplace=True)
    df = df.drop(0)
    df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

    print(f"Loaded {file_path}: {len(df)} rows, {len(df.columns)} columns")
    return df

#Bins- quartile
def make_ia_bins(df: pd.DataFrame, ia_col: str, n_bins: int = IA_BIN_COUNT) -> list[list[float]]:
    qs = [df[ia_col].quantile(p) for p in np.linspace(0, 1, n_bins + 1)]
    bins = [[lb, ub] for lb, ub in zip(qs[:-1], qs[1:])]
    bins[-1][1] = np.nextafter(bins[-1][1], float('inf'))  # include upper edge
    return bins

def bin_by_ia(df: pd.DataFrame, ia_bins: list[list[float]], ia_col: str, sa_col: str, fy_col: str, fz_col: str) -> list[pd.DataFrame]:
    out = []
    for lb, ub in ia_bins:
        sub = df[(df[ia_col] >= lb) & (df[ia_col] < ub)][[sa_col, fy_col, ia_col, fz_col]].dropna()
        out.append(sub)
    return out

def fit_pacejka_Fy_bin(data_bin: pd.DataFrame, sa_col: str, fy_col: str):
    x = data_bin[sa_col].values
    y = data_bin[fy_col].values
    if len(x) < 12 or np.nanstd(x) < 1e-6:
        return None
    D_guess = float(np.nanmax(np.abs(y))) if np.isfinite(y).any() else 1000.0
    p0 = [8.0, 1.3, max(D_guess, 100.0), 0.0, 0.0]  # [B, C, D, E, F]
    try:
        popt, _ = curve_fit(pacejka_lateral, x, y, p0=p0, maxfev=200000)
        return popt
    except Exception:
        return None

#Plot data
def plot_fy_over_sa_by_ia(ia_binned: list[pd.DataFrame], params_binned: list, ia_bins: list[list[float]],
                          sa_col: str, fy_col: str, fz_col: str) -> go.Figure:
    rows, cols = GRAPH_ROWS, GRAPH_COLS
    titles = [f"fy over SA for IA in bin [{lb:.2f}, {ub:.2f}) deg" for lb, ub in ia_bins]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

    # pick a fixed palette (Plotly built-in qualitative colors)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    nplots = min(len(ia_binned), rows * cols)
    for i in range(nplots):
        data_bin = ia_binned[i]
        if data_bin.empty:
            continue
        r = i // cols + 1
        c = i % cols + 1
        color = colors[i % len(colors)]

        # scatter points (uniform color per bin, no colorbar)
        fig.add_trace(
            go.Scatter(
                x=data_bin[sa_col],
                y=data_bin[fy_col],
                mode="markers",
                marker=dict(color=color, size=5, opacity=0.6),
                name=f"Data (IA bin {i+1})"
            ),
            row=r, col=c
        )

        # fitted curve (same color, solid line)
        if params_binned[i] is not None:
            sa_min, sa_max = float(np.min(data_bin[sa_col])), float(np.max(data_bin[sa_col]))
            if sa_min == sa_max:
                sa_min, sa_max = sa_min - 0.1, sa_max + 0.1
            sa_dom = np.linspace(sa_min, sa_max, 400)
            y_fit = pacejka_lateral(sa_dom, *params_binned[i])
            fig.add_trace(
                go.Scatter(x=sa_dom, y=y_fit, mode="lines", line=dict(color=color, width=2),
                           name=f"Pacejka fit (IA bin {i+1})"),
                row=r, col=c
            )

        fig.update_xaxes(title_text="Slip Angle SA (deg)", row=r, col=c)
        fig.update_yaxes(title_text="Lateral Force Fy (N)", row=r, col=c)

    fig.update_layout(
        title_text="Lateral Force (Fy) over Slip Angle SA for Camber bins (IA)",
        showlegend=False
    )
    return fig

# --- added: combined plot showing all IA bins together
def plot_combined_fy_over_sa(ia_binned: list[pd.DataFrame], params_binned: list, ia_bins: list[list[float]],
                             sa_col: str, fy_col: str) -> go.Figure:
    """
    Creates a single combined plot of Fy vs. SA across all IA bins,
    each in a different color with legend labels.
    """
    fig = go.Figure()

    # same fixed palette for consistency
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for i, (data_bin, params, bounds) in enumerate(zip(ia_binned, params_binned, ia_bins)):
        if data_bin.empty:
            continue
        color = colors[i % len(colors)]
        label = f"IA bin {i+1} [{bounds[0]:.2f}, {bounds[1]:.2f})°"

        # scatter points (each bin same color)
        fig.add_trace(go.Scatter(
            x=data_bin[sa_col],
            y=data_bin[fy_col],
            mode="markers",
            marker=dict(color=color, size=4, opacity=0.6),
            name=f"Data {label}"
        ))

        # fitted curve (same color, solid line)
        if params is not None:
            sa_min, sa_max = float(np.min(data_bin[sa_col])), float(np.max(data_bin[sa_col]))
            sa_dom = np.linspace(sa_min, sa_max, 400)
            y_fit = pacejka_lateral(sa_dom, *params)
            fig.add_trace(go.Scatter(
                x=sa_dom,
                y=y_fit,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"Pacejka fit {label}"
            ))

    fig.update_layout(
        title_text="Combined Lateral Force (Fy) over Slip Angle SA for Camber bins (IA)",
        xaxis_title="Slip Angle SA (deg)",
        yaxis_title="Lateral Force Fy (N)",
        legend=dict(
            x=1.02, y=1,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
            borderwidth=1,
            title="Legend"
        ),
        template="plotly_white"
    )
    return fig

def plot_time_vs_camber(df: pd.DataFrame, time_col: str, ia_col: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[time_col].astype(int),  #convert to integer
        y=df[ia_col],
        mode='lines+markers',
        line=dict(color='blue'),
        marker=dict(size=4),
        name='Camber Angle over Time'
    ))

    fig.update_layout(
        title='Camber Angle (IA) over Time',
        xaxis_title='Time (s)',
        yaxis_title='Camber Angle (IA) (deg)',
        template='plotly_white'
    )

    return fig


# Main 
if __name__ == "__main__":
    all_data = get_run_data()

    # Column names from your probe
    IA_COL = "IA deg"   # camber / inclination angle (deg)
    SA_COL = "SA deg"   # slip angle (deg)
    FY_COL = "FY N"     # lateral force (N)
    FZ_COL = "FZ N"     # normal load (N) (negative = compressive)

    # Build IA bins and slice data
    IA_bins = make_ia_bins(all_data, IA_COL, n_bins=IA_BIN_COUNT)
    IA_binned = bin_by_ia(all_data, IA_bins, IA_COL, SA_COL, FY_COL, FZ_COL)

    # Fit per IA bin
    params_binned_IA = [fit_pacejka_Fy_bin(df_bin, SA_COL, FY_COL) for df_bin in IA_binned]

    # Report fit quality
    for i, (bounds, df_bin) in enumerate(zip(IA_bins, IA_binned), start=1):
        print(f"IA Bin {i} [{bounds[0]:.3f}, {bounds[1]:.3f}) deg | n={len(df_bin)}")
        p = params_binned_IA[i-1]
        if p is None or df_bin.empty:
            print("  Fit failed or insufficient data.")
            continue
        y_hat = pacejka_lateral(df_bin[SA_COL].values, *p)
        print("  Params:", p)
        print("  R2:", r2_score(df_bin[FY_COL].values, y_hat))

    # Plot and export (correctly labeled)
    fig = plot_fy_over_sa_by_ia(IA_binned, params_binned_IA, IA_bins, SA_COL, FY_COL, FZ_COL)

    # timestamped file generation
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_path = f"htmlGraphs/fy_over_sa_by_ia_({timestamp}).html"

    fig.write_html(output_path)
    print(f"Wrote {output_path}")

    #full data 
    fig_combined = plot_combined_fy_over_sa(IA_binned, params_binned_IA, IA_bins, SA_COL, FY_COL)

    # timestamped combined file
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    combined_path = f"htmlGraphs/full_data_({timestamp}).html"

    fig_combined.write_html(combined_path)
    print(f"Wrote {combined_path}")


    #time vs camber (ia)
    figTimeCamber = plot_time_vs_camber(all_data, time_col="ET s", ia_col="IA deg")
    figTimeCamber.write_html(f"htmlGraphs/time_vs_camber_({timestamp}).html")
    print(f"Wrote htmlGraphs/time_vs_camber_({timestamp}).html")

    sys.exit(0)
