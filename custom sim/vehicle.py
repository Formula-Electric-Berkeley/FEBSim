# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:30:13 2023

@author: EJDRO
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def read_info(workbook_file, sheet_name=1, start_row=2, end_row=10000, cols="B:C"):
    # Setup the Import Options
    opts = pd.io.excel.read_excel(workbook_file, sheet_name, header=None, skiprows=start_row-1, nrows=end_row-start_row+1, usecols=cols)
    
    # Specify column names
    opts.columns = ["Variable", "Value"]
    
    return opts


filename = 'FEB_SN3_30kW.xlsx'

info = read_info(filename,'Info')
data = read_info(filename,'Torque Curve', cols="A:B")

#print(data)


#print(info)

i = 2

#mass
M = info.at[(i, "Value")]
i += 1
df = info.at[(i, "Value")]/100
i += 1

#wheelbase 
L = info.at[(i, "Value")]/1000
i += 1

# steering rack ratio
rack = info.at[(i, "Value")]
i += 1

# aerodynamics
Cl = info.at[(i, "Value")]
i += 1
Cd = info.at[(i, "Value")]
i += 1

factor_Cl = info.at[(i, "Value")]
i += 1
factor_Cd = info.at[(i, "Value")]
i += 1

da = info.at[(i, "Value")]/100
i += 1
A = info.at[(i, "Value")]
i += 1
rho = info.at[(i, "Value")]
i += 1


# brakes
br_disc_d = info.at[(i, "Value")]/1000
i += 1
br_pad_h = info.at[(i, "Value")]/1000
i += 1
br_pad_mu = info.at[(i, "Value")]
i += 1
br_nop = info.at[(i, "Value")]
i += 1
br_pist_d = info.at[(i, "Value")]
i += 1
br_mast_d = info.at[(i, "Value")]
i += 1
br_ped_r = info.at[(i, "Value")]
i += 1

# tyres
factor_grip = info.at[(i, "Value")]
i += 1
tyre_radius = info.at[(i, "Value")]/1000
i += 1
Cr = info.at[(i, "Value")]
i += 1
mu_x = info.at[(i, "Value")]
i += 1
mu_x_M = info.at[(i, "Value")]
i += 1
sens_x = info.at[(i, "Value")]
i += 1
mu_y = info.at[(i, "Value")]
i += 1
mu_y_M = info.at[(i, "Value")]
i += 1
sens_y = info.at[(i, "Value")]
i += 1
CF = info.at[(i, "Value")]
i += 1
CR = info.at[(i, "Value")]
i += 1

# engine
factor_power = info.at[(i, "Value")]
i += 1
n_thermal = info.at[(i, "Value")]
i += 1
fuel_LHV = info.at[(i, "Value")]
i += 1

# drivetrain
drive = info.at[(i, "Value")]
i += 1
shift_time = info.at[(i, "Value")]
i += 1
n_primary = info.at[(i, "Value")]
i += 1
n_final = info.at[(i, "Value")]
i += 1
n_gearbox = info.at[(i, "Value")]
i += 1
ratio_primary = info.at[(i, "Value")]
i += 1
ratio_final = info.at[(i, "Value")]
i += 1
ratio_gearbox = info.at[(i, "Value")]

#We don't have gears or NOG, so that part can be left empty
#We exclude gears from our Powertrain model, but if we somehow get gears in the future, it should add back fast



#Done importing data...
pi = np.pi

# Brake Model
br_pist_a = 0.25*br_nop*pi*(br_pist_d/1000)**2  # [m2]
br_mast_a = 0.25*pi*(br_mast_d/1000)**2  # [m2]
beta = tyre_radius/(br_disc_d/2-br_pad_h/2)/br_pist_a/br_pad_mu/4 # [Pa/N] per wheel
#TODO this is absolutely cursed notation; derive beta later
#a/b/c = a/(b*c)
phi = br_mast_a/br_ped_r*2 # [-] for both systems




# Powertrain Model

motor_speeds = data.loc[:, "Variable"] #rpm
motor_torques = data.loc[:, "Value"] #Nm
power_curve = motor_speeds*motor_torques*2*pi/60 # W


wheel_speed = motor_speeds/ratio_primary/ratio_gearbox/ratio_final
vehicle_speed = wheel_speed*2*pi/60*tyre_radius #The theoretical speed of the vehicle at various torques
wheel_torque = motor_torques*ratio_primary*ratio_gearbox*ratio_final*n_primary*n_gearbox*n_final
#We push inefficiency of the gears into the torque (is this a good assumption? TODO)

# minimum and maximum vehicle speeds
v_min = min(vehicle_speed)
v_max = max(vehicle_speed)

# new speed vector for fine meshing
dv = 0.5/3.6
vehicle_speed_fine = np.linspace(v_min,v_max, (int) ((v_max-v_min)/dv) )

# engine tractive force
engine_force =  wheel_torque/tyre_radius
fx_engine = np.interp(vehicle_speed_fine, vehicle_speed, engine_force) #interpolate to our finer mesh

vehicle_speed = vehicle_speed_fine #to fix any future dependencies

# adding values for 0 speed to vectors for interpolation purposes at low speeds
#TODO add back if needed; don't add anything until we see the reason
#vehicle_speed = [0;vehicle_speed] ;
#gear = [gear(1);gear] ;
#fx_engine = [fx_engine(1);fx_engine] ;

# final vectors
# engine speed
engine_speed = ratio_final*ratio_gearbox*ratio_primary*vehicle_speed_fine/tyre_radius*60/(2*pi)
# wheel torque
wheel_torque = fx_engine*tyre_radius
# engine torque
engine_torque = wheel_torque/ratio_final/ratio_gearbox/ratio_primary/n_primary/n_gearbox/n_final
engine_torque = wheel_torque/(ratio_final*ratio_gearbox*ratio_primary*n_primary*n_gearbox*n_final)

# engine power
engine_power = engine_torque*engine_speed*2*pi/60


# Force model

# gravitational constant
g = 9.81 
# drive and aero factors
if drive == 'RWD':
    factor_drive = (1-df)       # weight distribution
    factor_aero = (1-da)        # aero distribution
    driven_wheels = 2           # number of driven wheels
elif drive == 'FWD':
        factor_drive = df 
        factor_aero = da 
        driven_wheels = 2 
else: #AWD
    factor_drive = 1 
    factor_aero = 1 
    driven_wheels = 4

# Z axis
fz_mass = -M*g #this ignores bank and inclination of the track
fz_aero = 1/2*rho*factor_Cl*Cl*A*vehicle_speed**2
fz_total = fz_mass+fz_aero
fz_tyre = (factor_drive*fz_mass+factor_aero*fz_aero)/driven_wheels

# x axis
fx_aero = 1/2*rho*factor_Cd*Cd*A*vehicle_speed**2
fx_roll = Cr*abs(fz_total)
fx_tyre = driven_wheels*(mu_x+sens_x*(mu_x_M*g-abs(fz_tyre)))*abs(fz_tyre)




# GGV Map

# track data; for simplicity, we assume no bank and inclination for now
bank = 0
incl = 0 #in degrees!
# lateral tyre coefficients
dmy = factor_grip*sens_y
muy = factor_grip*mu_y
Ny = mu_y_M*g
# longitudinal tyre coefficients
dmx = factor_grip*sens_x
mux = factor_grip*mu_x
Nx = mu_x_M*g

# normal load on all wheels
Wz = M*g*np.cos(bank)*np.cos(incl)
# induced weight from banking and inclination
Wy = -M*g*np.sin(bank)
Wx = M*g*np.sin(incl)

# speed map vector
dv = 2
v = np.linspace(0 ,v_max, (int) ((v_max-v_min)/dv) )

# friction ellipse points
N = 45
# map preallocation
GGV = np.zeros((len(v),2*N-1,3))

xdata = np.zeros((len(v),2*N-1))
ydata = np.zeros((len(v),2*N-1))
zdata = np.zeros((len(v),2*N-1))

for i in range(len(v)):
    # aero forces
    Aero_Df = 1/2*rho*factor_Cl*Cl*A*v[i]**2
    Aero_Dr = 1/2*rho*factor_Cd*Cd*A*v[i]**2
    
    # rolling resistance
    Roll_Dr = Cr*abs(-Aero_Df+Wz)
    
    # normal load on driven wheels
    Wd = (factor_drive*Wz+(-factor_aero*Aero_Df))/driven_wheels
    # drag acceleration
    ax_drag = (Aero_Dr+Roll_Dr+Wx)/M
    # maximum lat acc available from tyres
    ay_max = 1/M*(muy+dmy*(Ny-(Wz-Aero_Df)/4))*(Wz-Aero_Df)
    # max long acc available from tyres
    ax_tyre_max_acc = 1/M*(mux+dmx*(Nx-Wd))*Wd*driven_wheels
    # max long acc available from tyres
    ax_tyre_max_dec = -1/M*(mux+dmx*(Nx-(Wz-Aero_Df)/4))*(Wz-Aero_Df) 
    # getting power limit from engine
    
    ax_power_limit = 1/M*np.interp(v[i], vehicle_speed_fine, fx_engine*factor_power)
    ax_power_limit = ax_power_limit*np.ones(N)
    # lat acc vector
    ay = ay_max*np.cos(np.linspace(0,2*pi,N))
    # long acc vector
    ax_tyre_acc = ax_tyre_max_acc*np.sqrt(1-(ay/ay_max)**2)             # friction ellipse    
    ax_acc = np.minimum(ax_tyre_acc,ax_power_limit)+ax_drag             # limiting by engine power
    ax_dec = ax_tyre_max_dec*np.sqrt(1-(ay/ay_max)**2)+ax_drag          # friction ellipse
    
    # saving GGV map
    GGV[i, :, 0] = np.concatenate((ax_acc, ax_dec[1:]))
    GGV[i,:,1] = np.concatenate((ay, np.flipud(ay[1:])))
    GGV[i,:,2] = v[i]*np.ones(2*N-1)    
        

def plotGGV(): 
    ax = plt.axes(projection='3d')
    
    # Data for three-dimensional scattered points
    zdata = GGV[:, :, 2]
    ax.scatter3D(GGV[:, :, 0], GGV[:, :, 1], zdata, c=zdata, cmap='Blues')
    
    ax.set_xlabel('Long acc [m/s^2]')
    ax.set_ylabel('Lat acc [m/s^2]')
    ax.set_zlabel('Speed [m/s]')
    
    plt.show()
    
#plotGGV()
        