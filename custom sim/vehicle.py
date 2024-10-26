# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:30:13 2023

@author: EJDRO
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import powertrain_model
motor = powertrain_model.motor()

power_cap = 80

pi = np.pi

# gravitational constant
g = 9.81

class Vehicle():
    def __init__(self, motor_data_file = "motor_curves/raw_curves.csv", filename = 'vehicle_files/FEB_SN3_30kW.xlsx', two_track = True):
        self.load_motor_data(motor_data_file)
        self.load_info(filename)
        if two_track:
            self.two_track_vars()
        #Done importing data...
        self.brake_model()
        self.powertrain_model()
        self.other_var_stuff()


    def load_motor_data(self, motor_data_file):
        motor_data = pd.read_csv(motor_data_file)
        self.motor_efficiency = np.median(motor_data["Motor Efficiency"])
        self.inverter_efficiency = np.median(motor_data["Inverter Efficiency"])
        self.max_raw_torque = np.max(motor_data["Original Peak Torque (Nm)"])

    # TODO: Do interpolation or something for motor efficiency
    def get_torque(self, rpm):
        '''
        RPM - Current rpm of the motor; 
        Function returns torque that can be applied
        '''
        cap = power_cap * 1000 # kW -> W
        angular_velocity = (rpm * 2 * np.pi) / 60 # rpm -> rad/s
        f = (cap * self.inverter_efficiency * self.motor_efficiency) / angular_velocity
        g = self.max_raw_torque
        return min(f, g)

    
    def read_info(self, workbook_file, sheet_name=1, start_row=2, end_row=10000, cols="B:C"):
        # Setup the Import Options
        opts = pd.io.excel.read_excel(workbook_file, sheet_name, header=None, skiprows=start_row-1, nrows=end_row-start_row+1, usecols=cols)

        # Specify column names
        opts.columns = ["Variable", "Value"]
        
        return opts

    def load_info(self, filename):
        self.info = self.read_info(filename,'Info')
        self.data = self.read_info(filename,'Torque Curve', cols="A:B")
        self.load_vars()

    # check implementation later
    def load_vars2(self):
        self.infoVars = {}
        for i in range(2, len(self.info)):
            self.infoVars[self.info.at[(i, "Variable")]] = self.info.at[(i, "Value")]
        print(self.infoVars)

    def load_vars(self):
        i = 2
        
        #mass
        self.M = self.info.at[(i, "Value")]
        i += 1
        self.df = self.info.at[(i, "Value")]/100
        i += 1

        #wheelbase 
        self.L = self.info.at[(i, "Value")]/1000
        i += 1

        # steering rack ratio
        self.rack = self.info.at[(i, "Value")]
        i += 1

        # aerodynamics
        self.Cl = self.info.at[(i, "Value")]
        i += 1
        self.Cd = self.info.at[(i, "Value")]
        i += 1

        self.factor_Cl = self.info.at[(i, "Value")]
        i += 1
        self.factor_Cd = self.info.at[(i, "Value")]
        i += 1

        self.da = self.info.at[(i, "Value")]/100
        i += 1
        self.A = self.info.at[(i, "Value")]
        i += 1
        self.rho = self.info.at[(i, "Value")]
        i += 1

        # brakes
        self.br_disc_d = self.info.at[(i, "Value")]/1000
        i += 1
        self.br_pad_h = self.info.at[(i, "Value")]/1000
        i += 1
        self.br_pad_mu = self.info.at[(i, "Value")]
        i += 1
        self.br_nop = self.info.at[(i, "Value")]
        i += 1
        self.br_pist_d = self.info.at[(i, "Value")]
        i += 1
        self.br_mast_d = self.info.at[(i, "Value")]
        i += 1
        self.br_ped_r = self.info.at[(i, "Value")]
        i += 1

        # tyres
        self.factor_grip = self.info.at[(i, "Value")]

        i += 1
        self.tyre_radius = self.info.at[(i, "Value")]/1000
        i += 1
        self.Cr = self.info.at[(i, "Value")]
        i += 1
        self.mu_x = self.info.at[(i, "Value")]
        i += 1
        self.mu_x_M = self.info.at[(i, "Value")]
        i += 1
        self.sens_x = self.info.at[(i, "Value")]
        i += 1
        self.mu_y = self.info.at[(i, "Value")]
        i += 1
        self.mu_y_M = self.info.at[(i, "Value")]
        i += 1
        self.sens_y = self.info.at[(i, "Value")]
        i += 1
        self.CF = self.info.at[(i, "Value")]
        i += 1
        self.CR = self.info.at[(i, "Value")]
        i += 1

        # engine
        self.factor_power = self.info.at[(i, "Value")]
        i += 1
        self.n_thermal = self.info.at[(i, "Value")]
        i += 1
        self.fuel_LHV = self.info.at[(i, "Value")]
        i += 1

        # drivetrain
        self.drive = self.info.at[(i, "Value")]
        i += 1
        self.shift_time = self.info.at[(i, "Value")]
        i += 1
        self.n_primary = self.info.at[(i, "Value")]
        i += 1
        self.n_final = self.info.at[(i, "Value")]
        i += 1
        self.n_gearbox = self.info.at[(i, "Value")]
        i += 1
        self.ratio_primary = self.info.at[(i, "Value")]
        i += 1
        self.ratio_final = self.info.at[(i, "Value")]
        i += 1
        self.ratio_gearbox = self.info.at[(i, "Value")]
    
    def two_track_vars(self):
        self.two_track = True
        self.drag_coeff = 1/2 * self.rho * self.factor_Cl * self.Cl * self.A #coefficient in front of v^2 for drag force calculation
        self.delta_max = 38 * np.pi / 180 #38 degrees is our max steering angle
        self.drive_max = 20000 #Nm
        self.brake_max = 50000 #N
        self.max_velocity = 30 #m/s; makes convergence faster

        #Brake coefficients (how is the brake force split up)
        self.brake_fr = 0.5*0.6
        self.brake_fl = 0.5*0.6
        self.brake_rl = 0.5*0.4
        self.brake_rr = 0.5*0.4

        self.Iz = 5550        # moment of inertiate
        self.lf = 1.0         # front wheelbase length
        self.lr = 1.2         # rear  wheelbase length
        self.wf = 1.7         # rear axle width
        self.wr = 1.7         # front axle width
        self.h  = 0.5         # distance from road surface to vehicle center of mass
        self.mu = 0.75        # coefficient of friction for tires
        self.tau_b = 0.05        # brake time constant (first order lag)

        self.Je = 0.28        # engine moment of inertia
        self.Jw = 0.05        # wheel moment of inertia
        self.R = 0.12055     # drivetrain gear ratio
        self.be = 0.008       # engine damping coefficient
        self.re = 0.35        # effective wheel radius
        self.rb = 0.3         # effective brake radius

        self.car_width = 0

        self.track_width_front = self.wf
        self.track_width_rear = self.wr
        self.cg_height = self.h

        #How to find the max rad/s of the wheels?
        self.gear_ratio = self.ratio_final
        self.omega_max = 5500/self.gear_ratio #5500 rpm is the max for the motor
        self.omega_max = self.omega_max*np.pi/30 #convert to rad/s
        self.omega_max = self.omega_max * 100 #factor of safety (to account for braking)
    
    def brake_model(self):
        self.br_pist_a = 0.25*self.br_nop*pi*(self.br_pist_d/1000)**2  # [m2]
        self.br_mast_a = 0.25*pi*(self.br_mast_d/1000)**2  # [m2]
        self.beta = self.tyre_radius/(self.br_disc_d/2-self.br_pad_h/2)/self.br_pist_a/self.br_pad_mu/4 # [Pa/N] per wheel
        #TODO this is absolutely cursed notation; derive beta later
        #a/b/c = a/(b*c)
        self.phi = self.br_mast_a/self.br_ped_r*2 # [-] for both systems

    def powertrain_model(self):
        self.motor_speeds = self.data.loc[:, "Variable"] #rpm
        self.motor_torques = self.data.loc[:, "Value"] #Nm
        self.power_curve = self.motor_speeds*self.motor_torques*2*pi/60 # W


        self.motor_affected_stuff()
        #TODO We push inefficiency of the gears into the torque (is this a good assumption?)
        #Wheel torque = motor torque * gear ratio
        #Look here for energy calcs
    
    def motor_affected_stuff(self):
        self.wheel_speed = self.motor_speeds/self.ratio_primary/self.ratio_gearbox/self.ratio_final
        self.vehicle_speed = self.wheel_speed*2*pi/60*self.tyre_radius #The theoretical speed of the vehicle at various torques
        self.wheel_torque = self.motor_torques*self.ratio_primary*self.ratio_gearbox*self.ratio_final*self.n_primary*self.n_gearbox*self.n_final

        self.v_min = min(self.vehicle_speed)
        self.v_max = max(self.vehicle_speed)

        # new speed vector for fine meshing
        self.dv = 0.5/3.6
        self.vehicle_speed_fine = np.linspace(self.v_min,self.v_max, (int) ((self.v_max-self.v_min)/self.dv) )

        # engine tractive force
        self.engine_force =  self.wheel_torque/self.tyre_radius
        self.fx_engine = np.interp(self.vehicle_speed_fine, self.vehicle_speed, self.engine_force) #interpolate to our finer mesh
        self.vehicle_speed = self.vehicle_speed_fine #to fix any future dependencies


        # adding values for 0 speed to vectors for interpolation purposes at low speeds
        self.vehicle_speed = np.insert(self.vehicle_speed, 0, 0)
        self.fx_engine = np.insert(self.fx_engine, 0, self.fx_engine[0])

        self.wheel_torque = self.fx_engine*self.tyre_radius

    def other_var_stuff(self):
        # engine speed
        self.engine_speed = self.ratio_final*self.ratio_gearbox*self.ratio_primary*self.vehicle_speed/self.tyre_radius*60/(2*pi)
        # engine torque
        self.engine_torque = self.wheel_torque/(self.ratio_final*self.ratio_gearbox*self.ratio_primary*self.n_primary*self.n_gearbox*self.n_final)

        # engine power
        self.engine_power = self.engine_torque*self.engine_speed*2*pi/60

        # drive and aero factors
        if self.drive == 'RWD':
            self.factor_drive = (1-self.df)       # weight distribution
            self.factor_aero = (1-self.da)        # aero distribution
            self.driven_wheels = 2           # number of driven wheels
        elif self.drive == 'FWD':
            self.factor_drive = self.df 
            self.factor_aero = self.da 
            self.driven_wheels = 2 
        else: #AWD
            self.factor_drive = 1 
            self.factor_aero = 1 
            self.driven_wheels = 4

        # Z axis
        self.fz_mass = -self.M*g #this ignores bank and inclination of the track
        self.fz_aero = 1/2*self.rho*self.factor_Cl*self.Cl*self.A*self.vehicle_speed**2
        self.fz_total = self.fz_mass+self.fz_aero
        self.fz_tyre = (self.factor_drive*self.fz_mass+self.factor_aero*self.fz_aero)/self.driven_wheels

        # x axis
        self.fx_aero = 1/2*self.rho*self.factor_Cd*self.Cd*self.A*self.vehicle_speed**2
        self.fx_roll = self.Cr*abs(self.fz_total)
        self.fx_tyre = self.driven_wheels*(self.mu_x+self.sens_x*(self.mu_x_M*g-abs(self.fz_tyre)))*abs(self.fz_tyre)

        # GGV Map

        # track data; for simplicity, we assume no bank and inclination for now
        self.bank = 0
        self.incl = 0 #in degrees!
        # lateral tyre coefficients
        self.dmy = self.factor_grip*self.sens_y
        self.muy = self.factor_grip*self.mu_y
        self.Ny = self.mu_y_M*g
        # longitudinal tyre coefficients
        self.dmx = self.factor_grip*self.sens_x
        self.mux = self.factor_grip*self.mu_x
        self.Nx = self.mu_x_M*g

        # normal load on all wheels
        self.Wz = self.M*g*np.cos(self.bank)*np.cos(self.incl)
        # induced weight from banking and inclination
        self.Wy = -self.M*g*np.sin(self.bank)
        self.Wx = self.M*g*np.sin(self.incl)

        # speed map vector
        self.dv = 2
        self.v = np.linspace(0 ,self.v_max, (int) ((self.v_max-self.v_min)/self.dv) )

        # friction ellipse points
        self.N = 45
        # map preallocation
        self.GGV = np.zeros((len(self.v),2*self.N-1,3))

        self.xdata = np.zeros((len(self.v),2*self.N-1))
        self.ydata = np.zeros((len(self.v),2*self.N-1))
        self.zdata = np.zeros((len(self.v),2*self.N-1))

        for i in range(len(self.v)):
            # aero forces
            self.Aero_Df = 1/2*self.rho*self.factor_Cl*self.Cl*self.A*self.v[i]**2
            self.Aero_Dr = 1/2*self.rho*self.factor_Cd*self.Cd*self.A*self.v[i]**2
            
            # rolling resistance
            self.Roll_Dr = self.Cr*abs(-self.Aero_Df+self.Wz)
            
            # normal load on driven wheels
            self.Wd = (self.factor_drive*self.Wz+(-self.factor_aero*self.Aero_Df))/self.driven_wheels
            # drag acceleration
            self.ax_drag = (self.Aero_Dr+self.Roll_Dr+self.Wx)/self.M
            # maximum lat acc available from tyres
            self.ay_max = 1/self.M*(self.muy+self.dmy*(self.Ny-(self.Wz-self.Aero_Df)/4))*(self.Wz-self.Aero_Df)
            # max long acc available from tyres
            self.ax_tyre_max_acc = 1/self.M*(self.mux+self.dmx*(self.Nx-self.Wd))*self.Wd*self.driven_wheels
            # max long acc available from tyres
            self.ax_tyre_max_dec = -1/self.M*(self.mux+self.dmx*(self.Nx-(self.Wz-self.Aero_Df)/4))*(self.Wz-self.Aero_Df) 
            # getting power limit from engine
            
            self.ax_power_limit = 1/self.M*np.interp(self.v[i], self.vehicle_speed, self.fx_engine*self.factor_power)
            self.ax_power_limit = self.ax_power_limit*np.ones(self.N)
            # lat acc vector
            self.ay = self.ay_max*np.cos(np.linspace(0,2*pi,self.N))
            # long acc vector
            self.ax_tyre_acc = self.ax_tyre_max_acc*np.sqrt(1-(self.ay/self.ay_max)**2)             # friction ellipse    
            self.ax_acc = np.minimum(self.ax_tyre_acc,self.ax_power_limit)+self.ax_drag             # limiting by engine power
            self.ax_dec = self.ax_tyre_max_dec*np.sqrt(1-(self.ay/self.ay_max)**2)+self.ax_drag          # friction ellipse
            
            # saving GGV map
            self.GGV[i, :, 0] = np.concatenate((self.ax_acc, self.ax_dec[1:]))
            self.GGV[i,:,1] = np.concatenate((self.ay, np.flipud(self.ay[1:])))
            self.GGV[i,:,2] = self.v[i]*np.ones(2*self.N-1)

    def plotGGV(self): 
        ax = plt.axes(projection='3d')
        
        # Data for three-dimensional scattered points
        self.zdata = self.GGV[:, :, 2]
        ax.scatter3D(self.GGV[:, :, 0], self.GGV[:, :, 1], self.zdata, c=self.zdata, cmap='Blues')
        
        ax.set_xlabel('Long acc [m/s^2]')
        ax.set_ylabel('Lat acc [m/s^2]')
        ax.set_zlabel('Speed [m/s]')
        
        plt.show()

    def plotMotorCurve(self): 
        ax = plt.axes()
        
        ax.scatter(self.vehicle_speed, self.wheel_torque)
        ax.set_xlabel('Vehicle Speed (m/s)')
        ax.set_ylabel('Wheel Torque (Nm)')
        
        #ax.scatter(motor_speeds, motor_torques)
        #ax.set_xlabel('Motor Speed (rpm)')
        #ax.set_ylabel('Motor Torque (Nm)')
        
        plt.show()

    #Temporary implementation of OpenAll: used for sweeping across multiple masses and motor torque curves
    def soft_reload(self, new_mass, power_cap, new_aero=[]):
        # Change the mass
        self.M = new_mass
        # Change the aero
        if len(new_aero) > 1:
            self.Cl = new_aero[0]
            self.Cd = new_aero[1]


        # Import the motor curve 
        self.motor_speeds, self.motor_torques = motor.get_motor_curve(power_cap) #rpm, Nm
        self.motor_affected_stuff()

        # alter variables directly affected by the motor curve



