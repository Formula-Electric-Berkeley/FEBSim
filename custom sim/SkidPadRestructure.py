import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib.pyplot as plt

# Import the Track class from track.py
from track import Track

def pacejka_model(alpha, params):
    """
    Calculates the lateral force using the Pacejka Magic Formula.
    """
    B = params['B']
    C = params['C']
    D = params['D']
    E = params['E']
    Sv = params['Sv']
    return D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha)))) + Sv

def select_pacejka_params_for_load(FZ, pacejka_params_for_bins):
    """
    Selects the appropriate Pacejka parameters based on vertical load (FZ).
    """
    bins = sorted(pacejka_params_for_bins.keys())
    for bin_range in bins:
        lower, upper = bin_range
        if lower < FZ <= upper:
            return pacejka_params_for_bins[bin_range]
    # If FZ does not fall into any bin, use the closest available parameters
    FZ_abs = abs(FZ)
    min_diff = float('inf')
    closest_params = None
    for bin_range in bins:
        bin_lower, bin_upper = bin_range
        bin_center = (bin_lower + bin_upper) / 2
        diff = abs(FZ_abs - abs(bin_center))
        if diff < min_diff:
            min_diff = diff
            closest_params = pacejka_params_for_bins[bin_range]
    return closest_params

def calculate_load_transfer(lateral_acc, h_cg, track_width, axle_mass):
    """
    Calculates the load transfer for a given axle based on lateral acceleration.
    """
    # Calculate the load transfer (in Newtons) due to lateral acceleration
    load_transfer = (axle_mass * lateral_acc * h_cg) / track_width
    return load_transfer

def create_motor_torque_function(motor_speed_data, torque_data):
    """
    Creates an interpolation function for motor torque.
    """
    torque_function = interp1d(motor_speed_data, torque_data, kind='linear', fill_value="extrapolate")
    return torque_function

def bicycle_model_dynamics(t, state, vehicle_params, track, torque_function):
    # Unpack vehicle parameters
    mass = vehicle_params['mass']
    h_cg = vehicle_params['h_cg']
    track_width = vehicle_params['track_width']
    aero_params = vehicle_params['aero_params']
    L_f = vehicle_params['L_f']
    L_r = vehicle_params['L_r']
    I_z = vehicle_params['I_z']
    wheelbase = vehicle_params['wheelbase']
    tire_radius = vehicle_params['tire_radius']
    gear_ratio = vehicle_params['gear_ratio']
    final_drive_ratio = vehicle_params['final_drive_ratio']
    pacejka_params_for_bins = vehicle_params['pacejka_params_for_bins']
    drivetrain_efficiency = vehicle_params['drivetrain_efficiency']
    max_lateral_acc = vehicle_params['max_lateral_acc']
    max_braking_g = vehicle_params['max_braking_g']

    # State contains [x, y, psi, vy, yaw_rate, v, s]
    x, y, psi, vy, yaw_rate, v, s = state

    g = 9.81
    rho = 1.225  # kg/m^3

    # Ensure the vehicle speed is not zero to avoid division by zero
    epsilon = 1e-5
    v = max(v, epsilon)

    # Aerodynamic parameters
    C_L, C_D, A = aero_params

    # Aerodynamic downforce (proportional to square of speed)
    downforce = 0.5 * rho * v**2 * C_L * A

    # Aerodynamic drag
    drag_force = 0.5 * rho * v**2 * C_D * A

    # Rolling resistance
    rolling_resistance_coefficient = 0.015  # Typical value
    rolling_resistance = rolling_resistance_coefficient * mass * g

    # Mass split equally for simplicity (front and rear axle)
    mass_front = mass * 0.5
    mass_rear = mass * 0.5

    # Distribute aerodynamic downforce (assuming aero_balance of 0.5)
    aero_balance = 0.5  # Adjust this based on front/rear downforce distribution
    downforce_front = downforce * aero_balance
    downforce_rear = downforce * (1 - aero_balance)

    # Total axle weights including downforce
    total_weight_front = mass_front * g + downforce_front
    total_weight_rear = mass_rear * g + downforce_rear

    # Lateral acceleration at the CG
    lateral_acc = v * yaw_rate

    # Load transfer due to lateral acceleration
    load_transfer_front = calculate_load_transfer(lateral_acc, h_cg, track_width, mass_front)
    load_transfer_rear = calculate_load_transfer(lateral_acc, h_cg, track_width, mass_rear)

    # Adjust vertical loads on the tires
    FZ_front = total_weight_front  # Adjust as per load transfer if needed
    FZ_rear = total_weight_rear

    # Get curvature at current position s
    track_length = track.x[-1]
    s_wrapped = s % track_length  # Wrap around if s exceeds track length
    curvature = track.curvature_function(s_wrapped)

    # Compute steering angle
    delta = np.arctan(wheelbase * curvature)

    # Compute slip angles
    alpha_f = delta - np.arctan((vy + L_f * yaw_rate) / (v + epsilon))
    alpha_r = -np.arctan((vy - L_r * yaw_rate) / (v + epsilon))

    # Select the appropriate Pacejka parameters based on vertical load
    pacejka_params_front = select_pacejka_params_for_load(FZ_front, pacejka_params_for_bins)
    pacejka_params_rear = select_pacejka_params_for_load(FZ_rear, pacejka_params_for_bins)

    # Calculate lateral forces using the Pacejka model
    FY_front = pacejka_model(alpha_f, pacejka_params_front)
    FY_rear = pacejka_model(alpha_r, pacejka_params_rear)

    # Longitudinal forces
    # For EV, motor speed is directly related to wheel speed
    wheel_speed = v / tire_radius  # rad/s

    # Motor speed (assuming direct drive or fixed gear ratio)
    motor_speed = wheel_speed * gear_ratio * final_drive_ratio * (60 / (2 * np.pi))  # rpm

    # Get motor torque
    motor_torque = torque_function(motor_speed)

    # Calculate tractive force at wheels
    tractive_force = (motor_torque * gear_ratio * final_drive_ratio * drivetrain_efficiency) / tire_radius

    # Total longitudinal force
    F_long = tractive_force - drag_force - rolling_resistance

    # **Improved Braking Logic**
    # Calculate maximum allowable speed based on lateral acceleration limit
    if np.abs(curvature) > epsilon:
        max_speed = np.sqrt(max_lateral_acc / np.abs(curvature))
    else:
        max_speed = vehicle_params['max_speed']  # Set to a high value or vehicle's max speed

    # Determine if braking is needed
    speed_error = v - max_speed
    if speed_error > 0:
        # Proportional controller to calculate required deceleration
        kp = 1.0  # Proportional gain (adjust as needed)
        required_deceleration = kp * speed_error
        # Limit deceleration to maximum braking deceleration
        deceleration = min(required_deceleration, g * max_braking_g)
        # Calculate braking force
        F_brake = mass * deceleration
        # Update total longitudinal force with braking
        F_long -= F_brake
    else:
        F_brake = 0  # No braking needed

    # Equations of motion
    x_dot = v * np.cos(psi) - vy * np.sin(psi)
    y_dot = v * np.sin(psi) + vy * np.cos(psi)
    psi_dot = yaw_rate
    vy_dot = (FY_front * np.cos(delta) + FY_rear) / mass - v * yaw_rate
    yaw_rate_dot = (L_f * FY_front * np.cos(delta) - L_r * FY_rear) / I_z
    v_dot = F_long / mass
    s_dot = v  # Rate of change of position along the track

    return [x_dot, y_dot, psi_dot, vy_dot, yaw_rate_dot, v_dot, s_dot]

def simulate_bicycle_model_on_track(vehicle_params, track, initial_conditions, torque_function):
    # Time vector
    total_simulation_time = vehicle_params['total_simulation_time']
    t_span = [0, total_simulation_time]
    t_eval = np.linspace(0, total_simulation_time, 1000)  # Time steps for evaluation

    # Create curvature function
    track.curvature_function = interp1d(track.x, track.r, kind='linear', fill_value='extrapolate')

    # Solve the system of differential equations
    solution = solve_ivp(
        fun=bicycle_model_dynamics,
        t_span=t_span,
        y0=initial_conditions,
        args=(vehicle_params, track, torque_function),
        t_eval=t_eval,
        method='RK45',
    )

    # Extract results
    x = solution.y[0]
    y = solution.y[1]
    psi = solution.y[2]
    vy = solution.y[3]
    yaw_rate = solution.y[4]
    speed = solution.y[5]
    s = solution.y[6]

    # Compute total distance traveled along the path
    distance_traveled = s

    results = pd.DataFrame({
        'Time (s)': solution.t,
        'X Position (m)': x,
        'Y Position (m)': y,
        'Heading Angle (rad)': psi,
        'Lateral Velocity (m/s)': vy,
        'Yaw Rate (rad/s)': yaw_rate,
        'Speed (m/s)': speed,
        'Distance Traveled (m)': distance_traveled
    })

    return results

# Example usage
if __name__ == "__main__":
    # Predefined Pacejka parameters for different load bins
    pacejka_params_for_bins = {
        (0, 4000): {'B': 10, 'C': 1.9, 'D': 1.0, 'E': 0.97, 'Sv': 0},
    }

    # Vehicle parameters
    vehicle_params = {
        'mass': 245,  # kg
        'h_cg': 0.25,  # meters
        'track_width': 1.05,  # meters
        'wheelbase': 1.55,  # meters
        'L_f': 1.55 / 2,  # Distance from CG to front axle
        'L_r': 1.55 / 2,  # Distance from CG to rear axle
        'I_z': 150,  # kg*m^2
        'aero_params': [2.06, 1.12, 1.0],  # C_L, C_D, A (example values)
        'pacejka_params_for_bins': pacejka_params_for_bins,
        'tire_radius': 0.2286,  # 18 inches in meters
        'gear_ratio': 1.0,  # For EV with direct drive
        'final_drive_ratio': 1.0,  # For EV with direct drive
        'drivetrain_efficiency': 0.95,  # 95% efficiency
        'max_lateral_acc': 1.5 * 9.81,  # Maximum lateral acceleration (1.5g)
        'max_braking_g': 1.0,  # Maximum braking deceleration (1g)
        'max_speed': 60,  # Maximum speed in m/s
        'total_simulation_time': 200,  # Total simulation time in seconds
    }

    # Load track data
    track = Track('Michigan_2022_AutoX.xlsx')

    # Motor torque data for EV (example data)
    motor_speed_data = np.array([0, 2000, 4000, 6000, 8000, 10000])  # rpm
    torque_data = np.array([250, 250, 250, 200, 150, 100])  # Nm

    # Create motor torque function
    torque_function = create_motor_torque_function(motor_speed_data, torque_data)

    # Initial conditions: [x, y, psi, vy, yaw_rate, v, s]
    initial_speed = 0.1  # m/s to avoid division by zero
    initial_conditions = [0, 0, 0, 0, 0, initial_speed, 0]  # Start from rest at s=0

    # Run the simulation
    results_df = simulate_bicycle_model_on_track(
        vehicle_params, track, initial_conditions, torque_function
    )

    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Time (s)'], results_df['Speed (m/s)'], label='Vehicle Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Vehicle Speed Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
