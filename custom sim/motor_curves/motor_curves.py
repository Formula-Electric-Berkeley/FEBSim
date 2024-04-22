import pandas as pd
import numpy as np

def read_info(workbook_file, sheet_name=1, start_row=2, end_row=1000, cols="A:E"):
    # Setup the Import Options
    opts = pd.io.excel.read_excel(workbook_file, sheet_name, header=None, skiprows=start_row-1, nrows=end_row-start_row+1, usecols=cols)

    # Specify column names
    opts.columns = ["Motor Speed (rpm)", "Original Peak Torque (Nm)", "Original Peak Power (kW)", "Motor Efficiency", "Inverter Efficiency"]

    return opts


# Track excel file selection
filename = 'Power Capped Motor Curves.xlsx'
info = read_info(filename,'Sheet1')

motor_speed = info.loc[:, "Motor Speed (rpm)"] #0 or NaN on straights, otherwise a float
motor_speed = np.nan_to_num(motor_speed)
motor_speed = motor_speed.astype(float)

peak_torque = info.loc[:, "Original Peak Torque (Nm)"] #0 or NaN on straights, otherwise a float
peak_torque = np.nan_to_num(peak_torque)
peak_torque = peak_torque.astype(float)

peak_power = info.loc[:, "Original Peak Power (kW)"] #0 or NaN on straights, otherwise a float
peak_power = np.nan_to_num(peak_power)
peak_power = peak_power.astype(float)

motor_efficiency = info.loc[:, "Motor Efficiency"] #0 or NaN on straights, otherwise a float
motor_efficiency = np.nan_to_num(motor_efficiency)
motor_efficiency = motor_efficiency.astype(float)

inverter_efficiency = info.loc[:, "Inverter Efficiency"] #0 or NaN on straights, otherwise a float
inverter_efficiency = np.nan_to_num(inverter_efficiency)
inverter_efficiency = inverter_efficiency.astype(float)

power_cap = 10 #kW

power_capped = np.min((peak_power, power_cap*inverter_efficiency*motor_efficiency), axis=0)
torque_capped = np.divide(power_capped, motor_speed)*1000/(2*np.pi/60)

header = ['Motor Speed (rpm)', 'Torque (Nm)']

motor_speed = np.vstack(motor_speed)
torque_capped = np.vstack(torque_capped)

df0 = np.concatenate((motor_speed, torque_capped), axis=1)
df1 = pd.DataFrame(df0)

# Create a Pandas Excel writer using XlsxWriter as the engine
print('motor_curve{}.xlsx'.format(power_cap))
writer = pd.ExcelWriter('motor_curve{}.xlsx'.format(power_cap), engine='xlsxwriter')

# Write the dataframe to a excel sheet
df1.to_excel(writer, sheet_name='Sheet1', index=False, header=header)
writer.close()