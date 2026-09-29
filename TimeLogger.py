from cmath import sqrt
import math
pi = math.pi
#import numpy as np

class TimeLogger:
    #CONST
    def __init__(self, initial_distance, initial_velocity, delta_time, initial_acceleration, final_distance):
        self.initial_distance = initial_distance
        self.initial_velocity = initial_velocity
        self.delta_time = delta_time
        self.initial_acceleration = initial_acceleration
        self.final_distance = final_distance 
        self.tire_radius = .2667
        self.driver_ratio = 5
        self.acc_est = 7.85
        self.cf = .71
        self.cf_long = 1.4
        self.A = 1.00
        self.crr = .02
        self.mass_with_driver = 300
        self.L_wheelbase = 1.55
        self.rear_bias = 0.55
        self.H_cg = 0.31
        
    #functions
    def get_motor_speed(self, current_speed):
        return current_speed / (2 * pi * self.tire_radius) * 2 * pi * self.driver_ratio

    def get_power(self, current_motor_torque, current_motor_speed):
        return current_motor_torque * current_motor_speed / 1000

    def get_drag(self, current_speed):
        return self.cf * self.A * (current_speed ** 2)

    def get_rolling_resistance(self):
        return self.mass_with_driver * 9.81 * self.crr

    def get_acceleration(self, current_get_motor_torque, current_get_rolling_resistance, current_get_drag):
        a = (current_get_motor_torque * self.driver_ratio - current_get_rolling_resistance - current_get_drag) / self.tire_radius / self.mass_with_driver
        return min(a, self.get_acc_new())

    def get_speed(self, initial_acceleration, initial_speed):
        return initial_acceleration * self.delta_time + initial_speed

    def get_acc_new(self):
        return self.cf_long * (self.rear_bias * self.mass_with_driver * 9.81 + self.get_load_transfer()) / self.mass_with_driver

    def get_load_transfer(self):
        return self.acc_est * self.H_cg * self.mass_with_driver / self.L_wheelbase

    my_list_motor_speed = [-1.00, .01, 5.31, 16.54, 27.77, 39.00, 50.23, 61.47, 72.70, 83.96, 95.16,
            106.40,117.63,128.86,139.39,150.62,161.85,173.09,184.32,195.55,206.78,218.02
            ,229.25,240.48,251.68,262.94,284.92,295.94,307.17,318.40,328.83,336.23,346.60
            ,357.72,368.95,380.18,391.41,402.64,413.18,424.50,435.64,446.87,458.10,469.24
            ,480.57,491.59,502.81,513.91,525.15,536.38,547.47,557.79,568.99]

    my_dict_motor_curve = {-1.00:231.00, 0.01:230.23, 5.31:230.22, 16.54:230.22, 27.77:230.22, 
            39.00:230.22, 50.23:230.22 , 61.47:230.22, 72.70:230.22, 83.96:230.33,
            95.16:230.22 , 106.40:230.22 , 117.63:230.22 , 128.86:213.46 , 139.39:198.97,
            150.62:185.49, 161.85:173.05, 173.09:161.98, 184.32:152.13, 195.55:143.41,
            206.78:135.65, 218.02:128.67, 229.25:122.37, 240.48:116.66, 251.68:111.45,
            262.94:106.69, 273.59:102.51, 284.92:98.45, 295.94:94.78, 307.17:91.31,
            318.40:88.11, 328.83:85.34, 336.23:83.46, 346.60:80.97, 357.72:78.47,
            368.95:76.11, 380.18:73.86, 391.41:71.74, 402.64:69.76, 413.18:67.98,
            424.50:66.18, 435.64:64.49, 446.87:62.89, 458.10:61.38, 469.24:59.98,
            480.57:58.64, 491.59:57.41, 502.81:56.22, 513.91:55.12, 525.15:54.04,
            536.38:53.00, 547.47:52.02, 557.79:51.17, 568.99:50.24}

    #function
    def get_motor_torque(self, motor_speed):
        closest_key = (min(self.my_list_motor_speed, key=lambda x:abs(x-motor_speed)))
        return self.my_dict_motor_curve.get(closest_key)

    def get_next_distance(self, initial_distance, initial_velocity, initial_acceleration):
        current_distance = initial_distance + (initial_velocity * self.delta_time) + (.5 * initial_acceleration * (self.delta_time**2))
        return current_distance
    
    
    def get_time_given_distance(self):
        distance = self.initial_distance
        velocity = self.initial_velocity
        acceleration = self.initial_acceleration
        current_time = 0
     
        while distance < self.final_distance:
            new_distance = self.get_next_distance(distance, velocity, acceleration) 
            new_velocity = self.get_speed(acceleration, velocity)
            current_time += self.delta_time
            new_acceleration = self.get_acceleration(self.get_motor_torque(self.get_motor_speed(new_velocity)), self.get_rolling_resistance(), self.get_drag(new_velocity))
            distance = new_distance
            velocity = new_velocity
            acceleration = new_acceleration
        print(current_time)
        return(current_time)

    def get_apex_speed(self, accel_y, turn_radius):
        return sqrt(accel_y * turn_radius)

    def accelerate(self, curr_velocity, accel_x):
        return sqrt(curr_velocity ** 2 + 2 * accel_x * self.delta_time)

    def deccelerate(self, curr_velocity, accel_x):
        return sqrt(curr_velocity ** 2 - 2 * accel_x * self.delta_time)
            
#example of the function below    
# newTimeLogger = TimeLogger(0, 0, .025, 9.45594300712411, 75)
newTimeLogger = TimeLogger(0, 0, .025, 9.16722909636295, 75)
newTimeLogger.get_time_given_distance()
