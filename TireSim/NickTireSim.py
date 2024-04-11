import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# Define the simplified Pacejka model function
def pacejka_model(alpha, B, C, D, E, F):
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))) + F)


# Define the path to the CSV file
data_path = '/Users/nicholaslemoff/Desktop/FEBSim/TireSim/tirecsvfiles/B1320run50.csv'

# Load the data, skipping the first row which contains units
data = pd.read_csv(data_path, skiprows=1)

# Convert relevant columns to numeric types, assuming 'deg' is slip angle, 'N.1' is lateral force, and 'N.2' is vertical load
data['SA'] = pd.to_numeric(data['deg'], errors='coerce')
data['FY'] = pd.to_numeric(data['N.1'], errors='coerce')
data['FZ'] = pd.to_numeric(data['N.2'], errors='coerce')

data['FY'] = data['FY'] * -1

# Define vertical load ranges for splitting the data
load_ranges = [(-1600, -1200), (-1200, -800), (-800, -400), (-400, 0)]

# Create a subplot figure with the number of load ranges
fig = make_subplots(rows=len(load_ranges), cols=1, subplot_titles=[f'Load Range {lr}' for lr in load_ranges])

# Process each load range
for i, (lower_bound, upper_bound) in enumerate(load_ranges, start=1):
    # Filter the data based on the current load range
    load_data = data[(data['FZ'] >= lower_bound) & (data['FZ'] < upper_bound)]

    # Prepare the data for curve fitting
    x_data = load_data['SA']
    y_data = load_data['FY']

    # Apply curve fitting to find the optimal coefficients for each load range
    initial_guess = [0.5, 1.2, max(load_data['FY']), 1, 0]
    optimal_parameters, covariance = curve_fit(pacejka_model, x_data, y_data, p0=initial_guess, maxfev=15000)

    # Generate model predictions over the range of slip angles for visualization
    x_model = np.linspace(x_data.min(), x_data.max(), 500)
    y_model = pacejka_model(x_model, *optimal_parameters)

    # Calculate R^2 accuracy for each fitted curve
    y_pred = pacejka_model(x_data, *optimal_parameters)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Add the scatter plot for actual data and the line plot for the fitted curve to the subplot
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=f'Actual Data {i}'), row=i, col=1)
    fig.add_trace(go.Scatter(x=x_model, y=y_model, mode='lines', name=f'Fitted Model {i}'), row=i, col=1)

    # Print the optimal parameters and R^2 value for the current load range
    print(f"Load Range {lower_bound} to {upper_bound}")
    print(f"Optimal Parameters:\nB: {optimal_parameters[0]}\nC: {optimal_parameters[1]}\nD: {optimal_parameters[2]}\nE: {optimal_parameters[3]}\nF: {optimal_parameters[4]}")
    print(f"R² Accuracy: {r_squared:.3f}\n")

# Enhancing the plot
fig.update_layout(height=1000, title_text="Pacejka Model Fitting by Load Range")

# Display the plot
fig.show()

