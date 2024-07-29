import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

class motor():
    def __init__(self):
        src_file_name = 'motor_curves/raw_curves.csv'

        # Read File
        info = pd.read_csv(src_file_name)

        # Separate data
        self.motor_speed = info["Motor Speed (rpm)"]
        self.peak_torque = info["Original Peak Torque (Nm)"]
        self.peak_power = info["Original Peak Power (kW)"]
        self.motor_efficiency = info["Motor Efficiency"]
        self.inverter_efficiency = info["Inverter Efficiency"]


        self.LAMBDA          = 0.05728   # PM Flux Linkage of the motor

        self.N_P             = 10        # Number of pole pairs in the motor

        self.RPM_TO_OMEGA    = 0.10472   # 2pi / 60


    def get_motor_curve(self, power_cap):
        # Calculate from data
        power_capped = np.min((self.peak_power, power_cap * self.inverter_efficiency * self.motor_efficiency), axis=0)
        torque_capped = np.divide(power_capped, self.motor_speed)*1000/(2*np.pi/60)

        
        return self.motor_speed, torque_capped
    
    def plot_motor_curve(self, power_cap):
        speed, torque = self.get_motor_curve(power_cap)
        plt.plot(speed, torque)
        plt.show()
    
    # Helper function to get the corresponding motor torque given a motor speed at our power cap 
    def get_torque_from_motor_speed(self, known_motor_speed, power_cap):
        motor_speeds, motor_torques = self.get_motor_curve(power_cap) #rpm, Nm
        interp_func = interp1d(motor_speeds, motor_torques)
        # Interpolating the motor torque at known_wheel_speed
        motor_torque_at_known_speed = interp_func(known_motor_speed)

        return motor_torque_at_known_speed

        

    def export_curve(self, power_cap):
        motor_speed, torque_capped = self.get_motor_curve(power_cap)

        # Compile data into DataFrame
        header = ['Motor Speed (rpm)', 'Torque (Nm)']
        motor_curve_df = pd.DataFrame(data=zip(motor_speed, torque_capped), columns=header)

        # Export as CSV
        file_name = f"capped_curve{power_cap}kW.csv"
        motor_curve_df.to_csv(file_name, sep=',', encoding='utf-8', index=False)


    # Iq current will be the same as phase current amplitude below base speed

    def calculate_Iq_current(self, torque_Nm):

        # sourced from https://isopack.blogspot.com/2017/08/fun-with-interior-permanent-magnet.html

        return torque_Nm * 2/ (3 * self.N_P * self.LAMBDA) 



    def calculate_back_emf(self, motor_rpm, torque_Nm):

        omega_rads = motor_rpm * self.RPM_TO_OMEGA

        mech_power_kW = omega_rads * torque_Nm / 1000

        estimated_kV = self.interpolate_Kv(mech_power_kW)

        if estimated_kV == 0:
            return 0

        return motor_rpm / estimated_kV 
    

    # Maybe we can use bezier curves / piecewise sigmoids here to stay closer to the datasheet points?

    # Not sure about the relationship between power and Kv, but it's probably not this simple lol

    # This might be something we can test / collect data for !!

    def interpolate_Kv(self, power_kW):

        # lagrange interpolation of K_v based on the following points (kW, Kv):

        # Peak Load: (104, 5.65), Nominal Load: (64, 7.85), No Load (0, 10.14)

        # Things to note: nominal load based on LC motor, No Load probably still doesn't mean no power

        return -(0.000184796 * power_kW**2) - (.0239543 * power_kW) + 10.14
    

    # calculate the base speed at our current accumulator voltage and power draw
    def calculate_base_speed(self, accumulator_voltage, power_W):
        estimated_kV = self.interpolate_Kv(power_W/1000)        # convert to kW
        if estimated_kV == 0:
            return 0
        
        # calculate the base speed of the motor in rpm
        base_speed = 60 * accumulator_voltage / (20*np.pi/estimated_kV)

        
        GEAR_RATIO = 3.53 #gear ratio of the car                (veh.ratio_final)
        TIRE_RADIUS = 0.266 #radius of the tire in meters       (veh.tyre_radius)

        # convert to linear speed of the vehicle in m/s
        base_speed = base_speed * 2*np.pi* TIRE_RADIUS/(GEAR_RATIO*60)

        return base_speed, estimated_kV
    
    


    def motor_efficiency(motor_speed, motor_torque):
        colors = ['red', 'orange', 'green', 'blue', 'violet']

        base_name = 'motor_curves\\'

        labels = [86, 90, 94, 95, 96]
        datasets = []
        for i, label in enumerate(labels):
            dfi = pd.read_csv(base_name+'{}.csv'.format(labels[i]), header=None)
            dfi.columns = ['motor_speed', 'motor_torque']
            dfi['motor_speed'] = dfi['motor_speed'] #* 2*np.pi/60 # convert to rads/s

            eff_frame = pd.DataFrame({'efficiency': [labels[i]]})

            dfi = pd.concat((dfi, eff_frame))
            
            datasets.append(dfi)

        df_all = pd.concat([dfi[0], dfi[1], dfi[2], dfi[3], dfi[4]])

        
        plt.figure(figsize=(10, 8))


        for df, color, label in zip(datasets, colors, labels):
            # Plotting original points]
            plt.scatter(df['X'], df['Y'], label=f'{label}', color=color, alpha=0.5)

        

        # Create a grid for interpolation
        grid_x, grid_y = np.mgrid[0:5001:100j, 0:251:100j]

        # Interpolate the efficiency values on the grid
        points = df_all[['motor_speed', 'motor_torques']].values
        values = df_all[['efficiency']].values

        #grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

        #plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap='viridis')
        #plt.colorbar(label='Efficiency')


        plt.title('Motor Efficiency')
        plt.xlabel('Motor Speed (rad/s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.grid(True)
        plt.show()



        #return griddata(points, values, (motor_speed, motor_torque), method='cubic')



    #motor_efficiency(2000, 150)

def test_motor():
    motor().plot_motor_curve(30)

    
#test_motor()

