import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Pacejka Model
def pacejka_model(alpha, B, C, D, E, F):
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))) + F)

# Load and preprocess data
def load_and_preprocess_data(data_path):
    column_names = ['ET', 'V', 'N', 'SA', 'IA', 'RL', 'RE', 'P', 'FX', 'FY', 'FZ', 'MX', 'MZ',
                   'NFX', 'NFY', 'RST', 'TSTI', 'TSTC', 'TSTO', 'AMBTMP', 'SR'] # Add 'SL' if you run 125.dat
    try:
        # Read data with whitespace separator using raw string to avoid escape sequence warnings
        data = pd.read_csv(data_path, sep=r'\s+', names=column_names, on_bad_lines='skip', engine='python')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Remove first two rows and reset index
    data = data.iloc[2:].reset_index(drop=True)

    # Convert columns to numeric and drop rows with missing values
    data = data.apply(pd.to_numeric, errors='coerce')
    initial_count = data.shape[0]
    data.dropna(inplace=True)
    final_count = data.shape[0]
    print(f"Dropped {initial_count - final_count} rows due to missing or non-numeric values.")
    return data

# Fit Pacejka model and plot results
def fit_pacejka_and_plot(data, load_ranges):
    # Create a 2x2 grid for plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()  # Flatten to 1D array for easy indexing

    for i, (lower_bound, upper_bound) in enumerate(load_ranges):
        # Filter data for current load range
        load_data = data[(data['FZ'] >= lower_bound) & (data['FZ'] < upper_bound)]
        x_data = load_data['SR']
        y_data = load_data['FX']
        
        if len(x_data) < 5:
            print(f"Not enough data in range {lower_bound} to {upper_bound} for curve fitting.")
            axs[i].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
            axs[i].set_title(f'Load Range {lower_bound} to {upper_bound}')
            axs[i].set_xlabel('Slip Ratio (SR)')
            axs[i].set_ylabel('Longitudinal Force (FX)')
            continue

        # Fit the model
        initial_guess = [0.75, 1.2, max(load_data['FX']), 1, 0]
        try:
            optimal_parameters, _ = curve_fit(pacejka_model, x_data, y_data, p0=initial_guess, maxfev=15000)
        except RuntimeError:
            print(f"Curve fitting did not converge for load range {lower_bound} to {upper_bound}.")
            axs[i].text(0.5, 0.5, 'Curve Fit Failed', ha='center', va='center')
            axs[i].set_title(f'Load Range {lower_bound} to {upper_bound}')
            axs[i].set_xlabel('Slip Ratio (SR)')
            axs[i].set_ylabel('Longitudinal Force (FX)')
            continue

        # Generate model predictions
        x_model = np.linspace(x_data.min(), x_data.max(), 500)
        y_model = pacejka_model(x_model, *optimal_parameters)

        # Calculate R²
        y_pred = pacejka_model(x_data, *optimal_parameters)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Plot actual data and fitted model
        ax = axs[i]
        ax.scatter(x_data, y_data, color='blue', label='Actual Data', alpha=0.6)
        ax.plot(x_model, y_model, color='red', label='Fitted Model')
        ax.set_title(f'Load Range {lower_bound} to {upper_bound}')
        ax.set_xlabel('Slip Ratio (SR)')
        ax.set_ylabel('Longitudinal Force (FX)')
        ax.legend()

        # Print results
        print(f"Load Range {lower_bound} to {upper_bound}")
        print(f"Optimal Parameters:")
        print(f"  B: {optimal_parameters[0]:.4f}")
        print(f"  C: {optimal_parameters[1]:.4f}")
        print(f"  D: {optimal_parameters[2]:.4f}")
        print(f"  E: {optimal_parameters[3]:.4f}")
        print(f"  F: {optimal_parameters[4]:.4f}")
        print(f"  R² Accuracy: {r_squared:.3f}\n")
    
    plt.tight_layout()
    plt.suptitle("Pacejka Model Fitting by Load Range", y=1.02)
    plt.subplots_adjust(top=0.92)  # Adjust top to make space for suptitle
    plt.show()

# Perform empirical μ analysis
def empirical_mu_analysis(original_data, load_ranges):
    data = original_data.copy()

    # Calculate absolute forces and mu
    data['FZ_abs'] = data['FZ'].abs()
    data['FX_abs'] = data['FX'].abs()
    data['mu'] = data['FX_abs'] / data['FZ_abs']

    # Remove zero FZ and unrealistic mu values
    data = data[data['FZ_abs'] != 0]
    print(f"\nData after removing rows with FZ_abs = 0: {data.shape[0]} rows remaining.")
    data = data[(data['mu'] > 0) & (data['mu'] < 2)]
    print(f"Data after filtering unrealistic μ values: {data.shape[0]} rows remaining.")

    # Bin the data
    bin_labels = [f"{lower}-{upper} N" for (lower, upper) in load_ranges]
    bins = [lower for (lower, upper) in load_ranges] + [load_ranges[-1][1]]
    data['Load_Bin'] = pd.cut(data['FZ'], bins=bins, labels=bin_labels, include_lowest=True)

    # Count data points per bin
    bin_counts = data['Load_Bin'].value_counts(sort=False)
    print("\nData Distribution Across Load Bins:")
    print(bin_counts)
    
    # Retain bins with sufficient data
    sufficient_bins = bin_counts[bin_counts >= 5].index
    data = data[data['Load_Bin'].isin(sufficient_bins)]
    print("\nBins with sufficient data (>=5 data points) retained for analysis.")

    # Calculate statistics
    analysis = data.groupby('Load_Bin', observed=True)['mu'].agg(
        count='count',
        mean='mean',
        median='median',
        std='std',
        min='min',
        max='max'
    ).reset_index()
    print("\nμ_empirical Analysis per Load Bin:")
    print(analysis)
    return analysis

# Main function
def main():
    data_path = 'scaled_B2356raw70.dat'  # Change this path if necessary
    load_ranges = [(-1600, -1200), (-1200, -800), (-800, -400), (-400, 0)]

    # Load data
    data = load_and_preprocess_data(data_path)
    if data is None:
        return
    
    # Fit and plot Pacejka model
    fit_pacejka_and_plot(data, load_ranges)

    # Perform empirical mu analysis
    analysis_df = empirical_mu_analysis(data, load_ranges)

if __name__ == "__main__":
    main()
