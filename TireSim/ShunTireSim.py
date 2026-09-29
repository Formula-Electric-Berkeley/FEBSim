import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from itertools import cycle

#Define the simplified Pacejka model function
def pacejka_model(alpha, B, C, D, E, F):
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))) + F)

#Define the path to the CSV file
data_path = "tirecsvfiles/B1320run125.csv"

#Load in data without including labeling in first row
data = pd.read_csv(data_path, skiprows=0)
data = data.drop(data.index[0])

#Convert into numerics
data["SR"] = pd.to_numeric(data["SR"], errors="coerce")
data["FX"] = pd.to_numeric(data["FX"], errors="coerce")
data["FZ"] = pd.to_numeric(data["FZ"], errors="coerce")
data["IA"] = pd.to_numeric(data["IA"], errors="coerce")

#Flip the sign of FX to match the coordinate system
data["FX"] = data["FX"] * -1

#Normal Force bins
load_ranges = [(-1600, -1200), (-1200, -800), (-800, -400), (-400, 0)]

#Camber Angle Bins
#Note: Bin 2.2 to 3.3 does not contain any data
camber_angle_bins = [(0, 1.1), (1.1, 2.2), (2.2, 3.3), (3.3, 4.4)]

#Scaling factor and calculate mu value
scaling_factor = 0.6 #Note: This should be changed based on scaling factor
data['mu'] = scaling_factor * data['FX'] / data['FZ']


#Get rid of any n/a data points
data['FZ'] = pd.to_numeric(data['FZ'], errors='coerce')
data['mu'] = pd.to_numeric(data['mu'], errors='coerce')
data = data.dropna()

#Set upper limit for coefficient of friction
mu_upper_limit = 5 
filtered_data = data[(data['mu'] <= mu_upper_limit)]

#Plot size selection
plt.figure(figsize=(10, 6))

#Scatter plot for coefficient of friction as a function of fz
plt.scatter(filtered_data['FZ'], filtered_data['mu'], alpha=0.5, label='Coefficient of Friction')


#Print out the plot
plt.xlabel('Normal Force |FZ| (N)')
plt.ylabel('Coefficient of Friction (μ)')
plt.title('Coefficient of Friction vs. Normal Force')
plt.legend()
plt.grid(True)
plt.show()

#Plot size selection
plt.figure(figsize=(12, 8))

#Set colors for data points
colors = cycle(plt.cm.tab10(np.linspace(0, 1, len(camber_angle_bins) * len(load_ranges))))

#Used to save pacejka coefficients for each camber bin and fz combination
pacejka_params_list = []

#Plot data points based on camber angles and load range values
for ca_lower, ca_upper in camber_angle_bins:
    for lower_bound, upper_bound in load_ranges:
        color = next(colors)  #Get next color from cycle

        #use current bin values
        subset = data[(data['FZ'] >= lower_bound) & (data['FZ'] < upper_bound) &
                      (data['IA'] >= ca_lower) & (data['IA'] < ca_upper)]
        
        if not subset.empty:
            x_data = subset['SR']
            y_data = subset['mu']
            
            # Plot data points
            plt.scatter(x_data, y_data, alpha=0.5, label=f'CA Bin {ca_lower}-{ca_upper}, Load {lower_bound}-{upper_bound}', color=color)

            # Fit and plot Pacejka model (optional, replace with polynomial if preferred)
            try:
                params, cov = curve_fit(pacejka_model, x_data, y_data, p0=[4.0, 4.0, 4.0, 4.0, 4.0], maxfev=20000)
                x_fit = np.linspace(x_data.min(), x_data.max(), 100)
                y_fit = pacejka_model(x_fit, *params)
                plt.plot(x_fit, y_fit, color=color)  #Use the same color for model fit
                pacejka_params_list.append(list(params))  
                print(f"Fit parameters for CA Bin {ca_lower}-{ca_upper}, Load {lower_bound}-{upper_bound}: {params}")
            except RuntimeError as e:
                print(f"Could not fit model for CA Bin {ca_lower}-{ca_upper}, Load {lower_bound}-{upper_bound}. Error: {e}")

#print out the data points for each camber angle and fz combination and the line of best fit
plt.xlabel('Slip Ratio (SR)', fontsize=14)
plt.ylabel('Coefficient of Friction (μ)', fontsize=14)
plt.title('Coefficient of Friction over Slip Ratio for Camber Angle and Load Bins with Pacejka Model Fits', fontsize=16)
plt.legend(loc='best', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()


#Create a grid for FZ and SR
FZ_dense = np.linspace(data['FZ'].min(), data['FZ'].max(), 100)
SR_dense = np.linspace(data['SR'].min(), data['SR'].max(), 100)
FZ_grid, SR_grid = np.meshgrid(FZ_dense, SR_dense)

#Holds aggregated mu values
mu_aggregated = np.zeros_like(FZ_grid)

#Pacejka model for each parameter set
for params in pacejka_params_list:
    mu_current = pacejka_model(SR_grid, *params) 
    mu_aggregated += mu_current

#Average out aggregated values
mu_aggregated /= len(pacejka_params_list)

#Plot the interpolated surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(FZ_grid, SR_grid, mu_aggregated, cmap='viridis')

#Set z-axis boundaries
ax.set_zlim(data['mu'].min(), data['mu'].max())

#plot graph
ax.set_xlabel('FZ')
ax.set_ylabel('SR')
ax.set_zlabel('mu')
plt.show()