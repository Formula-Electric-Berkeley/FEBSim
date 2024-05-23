import numpy as np
from scipy.interpolate import interp1d

g = 9.81


#increment or decrement j, mod j_max
def next_point(j, j_max, mode, tr_config='Closed'):
    j_next = None

    if mode == 1:  # acceleration
        if tr_config == 'Closed':
            if j == j_max - 1:
                j = j_max
                j_next = 0
            elif j == j_max:
                j = 0
                j_next = 1
            else:
                j = j + 1
                j_next = j + 1
        elif tr_config == 'Open':
            j = j + 1
            j_next = j + 1

    elif mode == -1:  # deceleration
        if tr_config == 'Closed':
            if j == 1:
                j = 0
                j_next = j_max
            elif j == 0:
                j = j_max
                j_next = j - 1
            else:
                j = j - 1
                j_next = j - 1
        elif tr_config == 'Open':
            j = j - 1
            j_next = j - 1

    return j_next, j



def vehicle_model_comb(veh, tr, v, v_max_next, j, mode):
    overshoot = False
    
    # Getting track data
    dx = tr.dx[j]
    r = tr.r[j]
    incl = tr.incl[j]
    bank = tr.bank[j]
    factor_grip = tr.factor_grip[j] * veh.factor_grip
    g = 9.81
    
    # Getting vehicle data based on mode
    if mode == 1:
        factor_drive = veh.factor_drive
        factor_aero = veh.factor_aero
        driven_wheels = veh.driven_wheels
    else:
        factor_drive = 1
        factor_aero = 1
        driven_wheels = 4
    
    # External forces
    M = veh.M
    Wz = M * g * np.cos(np.radians(bank)) * np.cos(np.radians(incl))
    Wy = -M * g * np.sin(np.radians(bank))
    Wx = M * g * np.sin(np.radians(incl))
    Aero_Df = 0.5 * veh.rho * veh.factor_Cl * veh.Cl * veh.A * v**2
    Aero_Dr = 0.5 * veh.rho * veh.factor_Cd * veh.Cd * veh.A * v**2
    Roll_Dr = veh.Cr * (-Aero_Df + Wz)
    Wd = (factor_drive * Wz + (-factor_aero * Aero_Df)) / driven_wheels
    
    # Overshoot acceleration
    ax_max = mode * (v_max_next**2 - v**2) / (2 * dx)
    ax_drag = (Aero_Dr + Roll_Dr + Wx) / M
    ax_track_limit = ax_max - ax_drag # same as ax_needed in OpenLap
    
    # Current lateral acceleration
    ay = v**2 * r + g * np.sin(np.radians(bank))
    
    # Tyre forces
    dmy = factor_grip * veh.sens_y
    muy = factor_grip * veh.mu_y
    Ny = veh.mu_y_M * g
    dmx = factor_grip * veh.sens_x
    mux = factor_grip * veh.mu_x
    Nx = veh.mu_x_M * g
    
    if np.sign(ay) != 0:
        ay_max = (1 / M) * (np.sign(ay) * (muy + dmy * (Ny - (Wz - Aero_Df) / 4)) * (Wz - Aero_Df) + Wy)
        if np.abs(ay / ay_max) > 1:
            #print("THIS SHOULD NOT HAPPEN")
            ellipse_multi = 0 #just a check to be safe
        else:
            ellipse_multi = np.sqrt(1 - (ay / ay_max)**2)
    else:
        ellipse_multi = 1
    
    # Calculating driver inputs
    if ax_track_limit >= 0:
        ax_tyre_max = (1 / M) * (mux + dmx * (Nx - Wd)) * Wd * driven_wheels
        ax_tyre = ax_tyre_max * ellipse_multi
        engine_force_func = interp1d(veh.vehicle_speed, veh.factor_power * veh.fx_engine)
        ax_power_limit = (1 / M) * engine_force_func(v)
        scale = min([ax_tyre, ax_track_limit]) / ax_power_limit
        tps = max([min([1, scale]), 0]) #possible check
        bps = 0
        ax_com = tps * ax_power_limit
    else:
        ax_tyre_max = -(1 / M) * (mux + dmx * (Nx - (Wz - Aero_Df) / 4)) * (Wz - Aero_Df)
        ax_tyre = ax_tyre_max * ellipse_multi
        fx_tyre = min([-ax_tyre, -ax_track_limit]) * M
        bps = max([fx_tyre, 0]) * veh.beta #again possible check
        tps = 0
        ax_com = -min([-ax_tyre, -ax_track_limit])
        #Check where the command is when the issue occurs
    
    # Final results
    ax = ax_com + ax_drag
    #print(v**2 + 2*mode*ax*dx)
    v_next = np.sqrt(v**2 + 2 * mode * ax * dx)
    #print(v_next)
    
    if tps > 0 and v / v_next >= 0.999:
        tps = 1
    
    # Checking for overshoot
    if v_next / v_max_next > 1: 
        #print(v_next, "   ", v_max_next)
        #print("Overshoooot")
        overshoot = True
        v_next = np.inf
        ax = 0
        ay = 0
        tps = -1
        bps = -1
    
    return v_next, ax, ay, tps, bps, overshoot        
        


def vehicle_model_lat(veh, tr, p):
    # Initialisation
    g = 9.81
    r = tr.r[p]
    incl = tr.incl[p]
    bank = tr.bank[p]
    factor_grip = tr.factor_grip[p] * veh.factor_grip
    
    # Vehicle data
    factor_drive = veh.factor_drive
    factor_aero = veh.factor_aero
    driven_wheels = veh.driven_wheels
    
    # Mass
    M = veh.M
    
    # Normal load on all wheels
    Wz = M * g * np.cos(np.radians(bank)) * np.cos(np.radians(incl))
    
    # Induced weight from banking and inclination
    Wy = -M * g * np.sin(np.radians(bank))
    Wx = M * g * np.sin(np.radians(incl))
    '''
    # Z-axis forces
    fz_mass = -M * g
    fz_aero = 0.5 * veh.rho * veh.factor_Cl * veh.Cl * veh.A * veh.vehicle_speed**2
    fz_total = fz_mass + fz_aero
    
    # X-axis forces
    fx_aero = 0.5 * veh.rho * veh.factor_Cd * veh.Cd * veh.A * veh.vehicle_speed**2
    fx_roll = veh.Cr * np.abs(fz_total)
    
    # Drag limitation
    idx = np.argmin(np.abs(veh.factor_power * veh.fx_engine + fx_aero + fx_roll + Wx))
    v_drag_thres = 0  # [m/s]
    v_drag = veh.vehicle_speed[idx] + v_drag_thres
    '''
    
    # Speed solution
    if r == 0:  # Straight (limited by engine speed limit or drag)
        v = veh.v_max
        tps = 1  # Full throttle
        bps = 0  # 0 brake
    else:  # Corner (may be limited by engine, drag, or cornering ability)
        # Initial speed solution
        D = -0.5 * veh.rho * veh.factor_Cl * veh.Cl * veh.A
        dmy = factor_grip * veh.sens_y
        muy = factor_grip * veh.mu_y
        Ny = veh.mu_y_M * g
        dmx = factor_grip * veh.sens_x
        mux = factor_grip * veh.mu_x
        Nx = veh.mu_x_M * g
        
        a = -np.sign(r) * dmy / 4 * D**2
        b = np.sign(r) * (muy * D + (dmy / 4) * (Ny * 4) * D - 2 * (dmy / 4) * Wz * D) - M * r
        c = np.sign(r) * (muy * Wz + (dmy / 4) * (Ny * 4) * Wz - (dmy / 4) * Wz**2) + Wy
        
        if a == 0:
            v = np.sqrt(-c / b)
        elif b**2 - 4 * a * c >= 0:
            root1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            root2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            
            if root1 >= 0:
                v = np.sqrt(root1)
                #print("Velocity 1: {}".format(v))
            elif root2 >= 0:
                v = np.sqrt(root2)
                #print("Velocity 2: {}".format(v))
            else:
                raise ValueError(f"No real roots at point index: {p}")
        else:
            raise ValueError(f"Discriminant < 0 at point index: {p}")
        
        # No drag????
        v = min([v, veh.v_max])
        
        # Adjusting speed for drag force compensation
        adjust_speed = True
        while adjust_speed:
            Aero_Df = 0.5 * veh.rho * veh.factor_Cl * veh.Cl * veh.A * v**2
            Aero_Dr = 0.5 * veh.rho * veh.factor_Cd * veh.Cd * veh.A * v**2
            Roll_Dr = veh.Cr * (-Aero_Df + Wz)
            Wd = (factor_drive * Wz + (-factor_aero * Aero_Df)) / driven_wheels
            
            ax_drag = (Aero_Dr + Roll_Dr + Wx) / M
            
            ay_max = np.sign(r) / M * (muy + dmy * (Ny - (Wz - Aero_Df) / 4)) * (Wz - Aero_Df)
            
            ay_needed = v**2 * r + g * np.sin(np.radians(bank))
            
            if ax_drag <= 0:
                ax_tyre_max_acc = 1 / M * (mux + dmx * (Nx - Wd)) * Wd * driven_wheels
                engine_force_func = interp1d(veh.vehicle_speed, veh.factor_power * veh.fx_engine)
                ax_power_limit = 1 / M * engine_force_func(v)
                ay = ay_max * np.sqrt(1 - (ax_drag / ax_tyre_max_acc)**2)
                ax_acc = ax_tyre_max_acc * np.sqrt(1 - (ay_needed / ay_max)**2)
                scale = min([-ax_drag, ax_acc]) / ax_power_limit
                tps = max([min([1, scale]), 0])              
                bps = 0
            else:
                ax_tyre_max_dec = -1 / M * (mux + dmx * (Nx - (Wz - Aero_Df) / 4)) * (Wz - Aero_Df)
                ay = ay_max * np.sqrt(1 - (ax_drag / ax_tyre_max_dec)**2)
                ax_dec = ax_tyre_max_dec * np.sqrt(1 - (ay_needed / ay_max)**2)
                fx_tyre = max([ax_drag, -ax_dec]) * M
                bps = max([fx_tyre, 0]) * veh.beta
                tps = 0
            
            if ay / ay_needed < 1:
                v = np.sqrt((ay - g * np.sin(np.radians(bank))) / r) - 1E-3
        
            else:
                adjust_speed = False
    
    return v, tps, bps


def other_points(i, i_max):
    i_rest = np.arange(0, i_max)
    i_rest = np.delete(i_rest, i)
    return i_rest





# simulate laptime for straight line acceleration
def simulate_accel(veh):
    v = veh.vehicle_speed # for interpolating GGV

    ax_acc = [] # create GGV
    for i in range(len(v)):
        # aero forces
        Aero_Df = 1/2*veh.rho*veh.factor_Cl*veh.Cl*veh.A*v[i]**2
        Aero_Dr = 1/2*veh.rho*veh.factor_Cd*veh.Cd*veh.A*v[i]**2
        
        # rolling resistance
        Roll_Dr = veh.Cr*abs(-Aero_Df+veh.Wz)
        
        # normal load on driven wheels
        Wd = (veh.factor_drive*veh.Wz+(-veh.factor_aero*Aero_Df))/veh.driven_wheels
        
        # drag acceleration
        ax_drag = (Aero_Dr+Roll_Dr+veh.Wx)/veh.M

        # max long acc available from tyres
        ax_tyre_max_acc = 1/veh.M*(veh.mux+veh.dmx*(veh.Nx-Wd))*Wd*veh.driven_wheels

        # getting power limit from engine
        ax_power_limit = 1/veh.M*veh.fx_engine[i]

        # long acc vector

        ax_acc.append(np.minimum(ax_tyre_max_acc,ax_power_limit)+ax_drag)             # limiting by engine power

    # numerical integration
    dt = 0.01 # s
    current_vehicle_speed = 0 # velocity of the vehicle, standing start
    current_vehicle_position = 0 # position of the vehicle
    accelerating = True
    time_elapsed = 0

    while accelerating:
        acceleration = np.interp(current_vehicle_speed, v, ax_acc) #interpolate our GGV
        
        current_vehicle_position += current_vehicle_speed*dt        # move forward
        current_vehicle_speed += acceleration*dt                    # update velocity
        time_elapsed += dt                                          # count time

        # 75 meter straight
        if current_vehicle_position >= 75:
            accelerating = False # stop

    # energy consumption can be estimated from 1/2 m v^2

    return time_elapsed, current_vehicle_speed


    