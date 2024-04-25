import pandas as pd
import numpy as np

# Parameters (These may need to be changed based on your requirements)
power_cap = 80 #kW
src_file_name = 'raw_curves.csv'

# Read File
info = pd.read_csv(src_file_name)

# Separate data
motor_speed = info["Motor Speed (rpm)"]
peak_torque = info["Original Peak Torque (Nm)"]
peak_power = info["Original Peak Power (kW)"]
motor_efficiency = info["Motor Efficiency"]
inverter_efficiency = info["Inverter Efficiency"]

# Calculate from data
power_capped = np.min((peak_power, power_cap * inverter_efficiency * motor_efficiency), axis=0)
torque_capped = np.divide(power_capped, motor_speed)*1000/(2*np.pi/60)

# Compile data into DataFrame
header = ['Motor Speed (rpm)', 'Torque (Nm)']
data_frame = pd.DataFrame(data=zip(motor_speed, torque_capped), columns=header)

# Export as CSV
file_name = f"capped_curve{power_cap}kW.csv"
data_frame.to_csv(file_name, sep=',', encoding='utf-8', index=False)

print(data_frame)