import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

### Pacejka model function ###
def pacejka_model(alpha, B, C, D, E, F):
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))) + F)

### Data Processing: 'SL' is not included for the raw50.dat file so remove/add if necessary ###
column_names = ['ET', 'V', 'N', 'SA', 'IA', 'RL', 'RE', 'P', 'FX', 'FY', 'FZ', 'MX', 'MZ',
                'NFX', 'NFY', 'RST', 'TSTI', 'TSTC', 'TSTO', 'AMBTMP', 'SR']
data_path = 'B2356raw50.dat' # FEBSim/TireSim/tiredatfiles/B1320run125.dat for other
data = pd.read_csv(data_path, delim_whitespace=True, names=column_names, on_bad_lines='skip')
data = data.iloc[2:].reset_index(drop=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)

# Define vertical load ranges for splitting the data
load_ranges = [(-1600, -1200), (-1200, -800), (-800, -400), (-400, 0)]
fig, axs = plt.subplots(len(load_ranges), 1, figsize=(10, len(load_ranges) * 5))

# Process each load range
for i, (lower_bound, upper_bound) in enumerate(load_ranges, start=1):
    # Filter the data based on the current load range
    load_data = data[(data['FZ'] >= lower_bound) & (data['FZ'] < upper_bound)]

    # Prepare the data for curve fitting
    x_data = load_data['SR']
    y_data = load_data['FX']

    # Check if there's enough data to perform curve fitting
    if len(x_data) < 5:
        print(f"Not enough data in load range {lower_bound} to {upper_bound} to perform curve fitting.")
        continue

    # Apply curve fitting to find the optimal coefficients for each load range
    initial_guess = [0.75, 1.2, max(load_data['FX']), 1, 0]
    try:
        optimal_parameters, covariance = curve_fit(pacejka_model, x_data, y_data, p0=initial_guess, maxfev=15000)
    except RuntimeError:
        print(f"Curve fitting did not converge for load range {lower_bound} to {upper_bound}.")
        continue

    # Generate model predictions over the range of slip ratios for visualization
    x_model = np.linspace(x_data.min(), x_data.max(), 500)
    y_model = pacejka_model(x_model, *optimal_parameters)

    # Calculate R^2 accuracy for each fitted curve
    y_pred = pacejka_model(x_data, *optimal_parameters)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Use the appropriate subplot
    ax = axs[i - 1] if len(load_ranges) > 1 else axs

    # Add the scatter plot for actual data and the line plot for the fitted curve to the subplot
    ax.scatter(x_data, y_data, color='blue', label=f'Actual Data {i}')
    ax.plot(x_model, y_model, color='red', label=f'Fitted Model {i}')

    # Enhance the subplot
    ax.set_title(f'Load Range {lower_bound} to {upper_bound}')
    ax.set_xlabel('Slip Ratio (SR)')
    ax.set_ylabel('Longitudinal Force (FX)')
    ax.legend()

    # Print the optimal parameters and R^2 value for the current load range
    print(f"Load Range {lower_bound} to {upper_bound}")
    print(f"Optimal Parameters:")
    print(f"B: {optimal_parameters[0]}")
    print(f"C: {optimal_parameters[1]}")
    print(f"D: {optimal_parameters[2]}")
    print(f"E: {optimal_parameters[3]}")
    print(f"F: {optimal_parameters[4]}")
    print(f"R² Accuracy: {r_squared:.3f}\n")

### Plot ###
plt.tight_layout()
plt.suptitle("Pacejka Model Fitting by Load Range", y=1.02)
plt.show()
