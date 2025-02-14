import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import casadi as ca

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

        motor_speeds = np.asarray(self.motor_speed)
        capped_torques = np.asarray(torque_capped)
        return motor_speeds, capped_torques
    
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

        base_name = 'motor_curves/'

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




def test_motor():
    motor().plot_motor_curve(30)

    motor_speeds, motor_torques = motor().get_motor_curve(power_cap=30)
    print(motor_torques)

    '''test_motor_speed = 1500  # Motor speed within the range
    torque_interp = ca.interpolant('torque_interpolated', 'linear', [motor_speeds], motor_torques)
    test_torque = torque_interp(test_motor_speed)
    print(f"Interpolated torque at {test_motor_speed} rpm: {test_torque}")

    plt.scatter(motor_speeds, motor_torques, color='red', label="Data Points")
    plt.xlabel('Motor Speed (rpm)')
    plt.ylabel('Torque')
    plt.legend()
    plt.show()'''

    
#test_motor()
    
class drexler_differential():
    def __init__(self):
        ramp_angles =       [30, 40, 45, 50, 60] # degrees
        lock_up_percents =  [88, 60, 51, 42, 29] # %

        # SN3 differential:
        # Acceleration ramp angle: 45
        # Braking ramp angle: 60

        self.forward_lock_up_percent = 0.51
        self.reverse_lock_up_percent = 0.29
        self.preload = 10                   # in Nm, the initial resistance to the diff opening up (without any applied torque)
        self.state = 'Open'                 # begin with the diff open
        self.direction = 'Accelerating'     # start in acceleration

    # Return the torque supplied by the differential for a given applied torque
    # Make this functionally dependent on wheel speed later
    # For now, we will be fully locked below the differential torque and fully open above
    # In reality, there's a fricitonal force that comes from applied torque and depends 2nd order on the difference in wheel speed
    def get_differential_torque(self, applied_torque):
        
        if self.direction == 'Accelerating':
            # Accelerating 
            differential_torque = self.forward_lock_up_percent*applied_torque
        else:
            # Braking
            differential_torque = self.reverse_lock_up_percent*np.abs(applied_torque)

        return min(differential_torque, self.preload)
    
    # Updates whether the rear axle is accelerating or decelerating
    def update_direction(self, left_wheel_acceleration):
        if left_wheel_acceleration > 0:
            self.direction = 'Accelerating'
        else:
            self.direction = 'Decelerating'


    # The differential controls how the applied torque is distributed between the driven wheels
    # In our model, this is entirely based on (a) the applied torque, and (b) the difference in wheel response torques
    # Since brake force is applied equally, the "wheel response torque" is just the torque due to tire forces
    def differentiate(self, left_wheel_response_torque, right_wheel_response_torque, applied_torque):
        
        response_delta = left_wheel_response_torque - right_wheel_response_torque 
        differential_torque = self.get_differential_torque(applied_torque)

        # Before differentiation occurs, the applied torque is supplied equally to both wheels
        left_applied_torque = applied_torque*0.5
        right_applied_torque = applied_torque*0.5
        
        if differential_torque <= self.preload:
            # If the applied torque is sufficiently small, we're essentially an open differential
            self.state = 'Open'

        elif response_delta > differential_torque:
            # The difference in grip torques exceeds what the diff can provide, we become partially locked
            self.state = 'Partially Locked'

            # if the left wheel has more grip, apply more torque to the left wheel and less torque to the right wheel
            # otherwise, apply more torque to the right wheel
            left_applied_torque     +=  0.5*differential_torque*np.sign(response_delta)
            right_applied_torque    -= 0.5*differential_torque*np.sign(response_delta)

        else:
            # If the differential torque = or exceeds the response delta, we can fully negate it
            # To do this, apply an additional 0.5 * abs(response_delta) to the wheel with more grip
            # Apply - 0.5* abs(response_delta) to the wheel with less grip. This gives the wheels the same net torque 

            left_applied_torque     +=  0.5*response_delta
            right_applied_torque    -= 0.5*response_delta



        # The differential changes the driven torque at the wheels and does not directly change the grip
        return left_applied_torque, right_applied_torque
    

    '''
    Differential implementation
    - If open, the wheel speeds are independent, and both are given an equal amount of torque (1/2 applied)
    - If partially locked, the differential applies more torque to the wheel with more traction (more response torque)  
    - If fully locked, the driven wheel speeds are equal; the net torques must also be equal in steady state
        
    Generally, the goal of the differential is to decrease the torque applied to the faster spinning wheel (i.e. the wheel with less grip = response torque)

    Can- The angular speeds of the wheels average out to the angular speed at the input of the diff
    To determine the motor speed, we literally only need the wheel speeds; that's awesome for implementing the motor efficiency curve

    If we're accelerating, the frictional force at the wheels points forward
    If we're decelerating, the frictional force at the wheels points backward
    '''