import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to remove outliers using a rolling median
def remove_outliers(df, window_size, threshold_factor):
    rolling_median = df['Y'].rolling(window=window_size, center=True).median()
    deviation = abs(df['Y'] - rolling_median)
    threshold = threshold_factor * deviation.median()
    outliers = deviation > threshold
    return df[~outliers]

# Polynomial fitting function
def fit_poly(df):
    # Fit a polynomial curve of degree 'n' to the data
    # Change the degree if needed
    n = 7
    coefficients = np.polyfit(df['X'], df['Y'], n)
    p = np.poly1d(coefficients)

    # Print the equation
    equation = 'y = '
    equation += ' + '.join([f'{coeff:.2e}*x^{i}' for i, coeff in enumerate(coefficients[::-1])])
    #print(equation)

    # Return the fitting polynomial function
    return p

# Load the data; replace the string with the correct path if the CSV files are not in the same directory
base_name = 'battery_info/Energus/SonyVTC6_'


# Parameters for outlier removal
window_size = 30  # Adjust if necessary
threshold_factor = 3  # Adjust if necessary

# List of the constant current curves we have
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'magenta', 'pink', 'brown', 'black']
labels = ['0.5A', '1A', '2A', '3A', '5A', '7A', '10A', '15A', '20A', '30A']
datasets = []


#TODO: 3A is bad

# to extend the fit 
pull_point = pd.DataFrame({'X': [3.0], 'Y': [2.5]})

for i, label in enumerate(labels):
    Ai = pd.read_csv(base_name+'{}.csv'.format(label), header=None)

    # Add headers; the x 1-SOC in amp hours, y is voltage of cell
    Ai.columns = ['X', 'Y']

    # Remove outliers from each dataframe
    Ai = remove_outliers(Ai, window_size, threshold_factor)

    # Appending the new point
    Ai = pd.concat([Ai, pull_point], ignore_index=True)

    Ai['X'] = Ai['X'] * 1000 # convert to mAh

    

    datasets.append(Ai)


plt.figure(figsize=(10, 8))


def plot():
    for df, color, label in zip(datasets, colors, labels):
        # Fit polynomial (Assuming fit_poly is previously defined and working correctly)
        p = fit_poly(df)

        # Plotting original points
        plt.scatter(df['X'], df['Y'], label=f'Original {label}', color=color, alpha=0.5)

        # Plotting the fit
        
        max_x = 3000 # df['X'].max()

        X_fit = np.linspace(df['X'].min(), max_x, 500)
        Y_fit = p(X_fit)
        plt.plot(X_fit, Y_fit, label=f'Fit {label}', color=color)

    # Set specific parameters for the y-axis and x-axis
    plt.ylim(2.8, 4.2)  # Set the limits for the y-axis
    plt.yticks(np.arange(0, 4.3, 0.2))  # Set the ticks on the y-axis

    plt.xlim(0, 3250)  # Set the limits for the x-axis
    plt.xticks(np.arange(0, 3350, 250))  # Set the ticks on the x-axis

    plt.title('Battery Discharge Curves with Polynomial Fitting')
    plt.xlabel('Capacity (mAh)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    plt.show()

def return_polynomials():
    polynomials = []
    for df in datasets:

        # Fit polynomial (Assuming fit_poly is previously defined and working correctly)
        p = fit_poly(df)
        polynomials.append(p)

    return polynomials
    
plot()

plt.savefig('energus_plot.png')
