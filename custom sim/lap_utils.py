import numpy as np
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
    ax_track_limit = ax_max - ax_drag
    
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
        ax_power_limit = (1 / M) * np.interp(v, veh.vehicle_speed, veh.factor_power * veh.fx_engine)
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
    
    if tps > 0 and v / v_next >= 1:
        tps = 1
    
    # Checking for overshoot
    if v_next / v_max_next > 1.0001: #don't compare floats exactly, get a bit of leeway
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
    
    # Speed solution
    if r == 0:  # Straight (limited by engine speed limit or drag)
        v = min([veh.v_max, v_drag])
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
        
        v = min([v, veh.v_max, v_drag])
        
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
                ax_power_limit = 1 / M * np.interp(v, veh.vehicle_speed, veh.factor_power * veh.fx_engine)
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







'''
UNUSED: Solves for the velocity and acceleration at the next mesh point
'''
def updateKinematics(veh, tr, v, v_max_next, j, mode):
    #v is current velocity, and j is index of current mesh point
    #v_max_next is here to ensure we don't overshoot the velocity of the next mesh point
    #mode is whether we are accelerating (1) or decelerating (-1) -> not boolean in case we want to add more cases
    
    
    #Track data <- import from csv file like OpenLap
    dx = tr.dx[j]
    r = tr.r[j]
    incl = tr.incl[j] *np.pi/180 #inclination angle is given in degrees
    bank = tr.bank[j] *np.pi/180
    factor_grip = tr.factor_grip[j]*veh.factor_grip #total grip factor
    
    factor_drive = veh.factor_drive
    factor_aero = veh.factor_aero
    driven_wheels = veh.driven_wheels
    
    
    # Forces on the vehicle
    
    M = veh.M
    Wz = M*g*np.cos(bank)*np.cos(incl)     # gravitational force normal to the track
    
    #Lateral (y) and longitudinal (x) components of gravitational force
    Wy = -M*g*np.sin(bank)
    Wx = M*g*np.sin(incl)
    
    
    # aero forces (simple for now)
    downforce = 1/2*veh.rho*veh.factor_Cl*veh.Cl*veh.A*v**2
    drag = 1/2*veh.rho*veh.factor_Cd*veh.Cd*veh.A*v**2
    
    # rolling resistance -> from OpenLap; TODO: derive this
    rolling_resistance = veh.Cr*(-downforce+Wz) #this is effectively a drag force
    
    # normal load on driven wheels -> from OpenLap; TODO: derive this; in notes
    Wd = (factor_drive*Wz+(-factor_aero*downforce))/driven_wheels 
    
    
    
    #Current lateral acceleration (centripetal acceleration formula)
    ay = v**2*r+g*np.sin(bank) #we must have this acceleration in the y to corner the turn of radius r at velocity v
    
    
    
    
    '''
    Process: 
        ay is fixed by the track. Friction ellipse is fixed by the vehicle
        To stay on the friction ellipse, we usually need to input some ax (either throttle or braking)
        That said, we cannot input so much ax that we overshoot the v_max of the next mesh point and possibly spin out
        
    '''
    
    ax_max = mode*(v_max_next**2-v**2)/(2*dx) #from kinematic eqn; mode = -1 means we accelerate in -x direction
    #This is the absolute max ax we can do without overshooting the next mesh point
    
    # drag forces (and gravity) will bleed some acceleration, allowing us to command more than we otherwise could
    ax_drag = (drag+rolling_resistance+Wx)/M
    # ovesrhoot acceleration limit
    ax_track_limit = ax_max-ax_drag # this is a limitation of the track and vehicle; max ax we can go to not overshoot
    

    #Include friction ellipse here; this will scale ax
    # calculate what we need to maximize ax while maintaining ay

    # lateral tyre coefficients
    dmy = factor_grip*veh.sens_y  #We assume muy is linear in Fz
    muy = factor_grip*veh.mu_y
    Ny = veh.mu_y_M*g #Normal force when muy(Fz) = muy
    
    # longitudinal tyre coefficients
    dmx = factor_grip*veh.sens_x
    mux = factor_grip*veh.mu_x
    Nx = veh.mu_x_M*g
    
    # friction ellipse multiplier
    if ay == 0: #in a straight, no compensation needed for bank angle
        ellipse_multi = 1
        
    else: #cornering or compensating
    
        # max lat acc available from tyres
        ay_max = 1/M*(np.sign(ay)*(muy+dmy*(Ny-(Wz-downforce)/4))*(Wz-downforce)+Wy)
        
        # max combined long acc available from tyres
        if abs(ay/ay_max) > 1: #overshoot; should not happen
            ellipse_multi = 0
        else:
            ellipse_multi = np.sqrt(1-(ay/ay_max)**2)  # friction ellipse
    
    
    
    #We want to target ax = ax_track_limit if we can to accelerate as fast as possible
    
    
    
    if ax_track_limit >=0: # we need power from the motor
        
        # this is the max ax we can get from driven tyres
        ax_tyre_max = 1/M*(mux+dmx*(Nx-Wd))*Wd*driven_wheels #from F_max acc in notes
       
        # correct for friction ellipse
        ax_tyre = ax_tyre_max*ellipse_multi
        
        # getting power limit from engine************************************ fix interp
        #TODO develop an engine model 
        ax_power_limit = 1/M*np.interp(v, veh.vehicle_speed, veh.factor_power*veh.fx_engine)
        
        
        # final longitudinal acc command
        ax_com = min(ax_tyre, ax_track_limit)
        
        
        # getting tps value
        tps = abs(ax_com/ax_power_limit)
        bps = 0 #Full acceleration -> 0 breaking
        
        
    else: # we need to brake
        
        # max pure long acc available from *all* tyres
        ax_tyre_max = -1/M*(mux+dmx*(Nx-(Wz-downforce)/4))*(Wz-downforce)
        
        # max comb long acc available from all tyres (i.e. corrected for friction ellipse)
        ax_tyre = ax_tyre_max*ellipse_multi
        
        #both accelerations are negative; we want positive acc
        ax_tyre = -ax_tyre
        ax_track_limit = -ax_track_limit
        
        
        # tyre braking force
        # we are limited by the smaller of the two acceleration caps 
        fx_tyre = min(ax_tyre, ax_track_limit)*M
        
        # getting brake input
        bps = abs(fx_tyre)*veh.beta # make sure bps is positive
        tps = 0 # all brake, no bake
        
        # final long acc command, again negative
        ax_com = -min(ax_tyre, ax_track_limit)
        
        
    # final results
    
    # total vehicle longitudinal acc
    ax = ax_com + ax_drag
    
    # next speed value
    v_next = np.sqrt(v**2+2*mode*ax*tr.dx[j]) #assume constant acc between mesh points
    
    # correcting tps for full throttle when at v_max on straights
    if tps > 0 and v/v_next >= 1:
        tps = 1
    
    
    # checking for overshoot
    overshoot = False #assume false by default
    
    if v_next/v_max_next>1:
        overshoot = True
        # resetting values for overshoot
        v_next = np.inf
        ax = 0
        ay = 0
        tps = -1
        bps = -1    
    
    
    return (v_next, ax, ay, tps, bps, overshoot)
    #We solve for velocity at the next mesh point, as well as the current acceleration and drive commands
    
    
"""
UNUSEDThis is to solve for the maximum velocities at each mesh point given vehicle data
"""
def vehicle_lateral_solver(veh, tr, p):
    #veh = vehicle file
    #tr = track file  (gives radii, inclination, and bank at each mesh point)
    #p = mesh point to solve for
    
    #Track data
    g = 9.81
    r = tr.r[p] #get the corner radius at the given mesh point for the track
    #print(r)
    incl = tr.incl[p] *np.pi/180 #inclination angle is given in degrees
    bank = tr.bank[p] *np.pi/180
    
    factor_grip = tr.factor_grip[p]*veh.factor_grip #total grip factor
    
    #Vehicle data
    factor_drive = veh.factor_drive
    factor_aero = veh.factor_aero
    driven_wheels = veh.driven_wheels
    M = veh.M
    
    #Weight forces
    Wz = M*g*np.cos(bank)*np.cos(incl)     # normal load on all wheels
    Wy = -M*g*np.sin(bank)                 # induced weight from banking and inclination
    Wx = M*g*np.sin(incl)
    
    # z axis forces
    fz_mass = -M*g
    fz_aero = 0.5*veh.rho*veh.factor_Cl*veh.Cl*veh.A*veh.vehicle_speed**2 #downforce
    fz_total = fz_mass+fz_aero
    
    
    #right now, we make an array of forces for each vehicle speed
    
    # x axis forces
    fx_aero = 0.5*veh.rho*veh.factor_Cd*veh.Cd*veh.A*veh.vehicle_speed**2 #drag
    fx_roll = veh.Cr*abs(fz_total) #rolling resistance is just Cr * the total normal force
    
    # drag limitation
    idx = np.argmin(np.abs(veh.factor_power*veh.fx_engine+fx_aero+fx_roll+Wx)) 
    #TODO: what is idx? -> velocity at which point the sum of the forces in X is minimal (equilibrium)
    v_drag_thres = 0 # customization factor [m/s]
    v_drag = veh.vehicle_speed[idx]+v_drag_thres 
    
    
    
    
    #solve for velocity
    if r==0 or r != r: # we're on a straight (engine- or drag- limited) (r = 0 or NaN)
    #r = 1/R, which is the actual circular radius of each arc for straights, R = inf, so r = 0

        v = min(veh.v_max, v_drag) #singular velocity, not an array
        tps = 1 # full throttle
        #we can full throttle on all straights because the final v is the minimum of each segment (so we decel before curves)
        bps = 0 # 0 brake; could be useful for autonomous
        
    else: # (engine-, drag-, or corner- limited); cornering ability determined by downforce and friction
        # initial speed solution 
        #(this speed will give the necessary downforce to corner, but drag may not allow it, so we must adjust later)
        #^just like the v_comb algorithm
        
        # we assume pure lateral acceleration, so we have no longitudinal force to counteract drag
        # we need to reduce this initial v by a small amount so that we can give some longitudinal force while 
        # staying on the friction ellipse
        
        
        # downforce coefficient
        D = -0.5*veh.rho*veh.factor_Cl*veh.Cl*veh.A
        
        # lateral tyre coefficients
        dmy = factor_grip*veh.sens_y
        muy = factor_grip*veh.mu_y
        Ny = veh.mu_y_M*g
        
        # longitudinal tyre coefficients
        dmx = factor_grip*veh.sens_x
        mux = factor_grip*veh.mu_x
        Nx = veh.mu_x_M*g
        
        
        # 2nd degree polynomial coefficients ( a*x^2+b*x+c = 0 ); derived in notebook
        
        a = -np.sign(r)*dmy/4*D**2
        b = np.sign(r)*(muy*D+(dmy/4)*(Ny*4)*D-2*(dmy/4)*Wz*D)-M*r
        c = np.sign(r)*(muy*Wz+(dmy/4)*(Ny*4)*Wz-(dmy/4)*Wz**2)+Wy #sign error on this Wy in notes?
        descriminant = b**2-4*a*c
        # calculating polynomial roots to solve for v
        if a==0:
            v = np.sqrt(-c/b)
            
        elif descriminant >=0:
            solution = (-b+np.sqrt(descriminant))/(2*a)
            if (solution >=0): #try positive solution
                v = np.sqrt(solution)
                         
            elif (-b-np.sqrt(descriminant))/(2*a) >=0: #try negative solution
                v = np.sqrt((-b-np.sqrt(descriminant))/(2*a))
                         
            else:
                print('No real roots at point index: {}'.format(p))
            
        else:
            print('Discriminant <0 at point index: {}'.format(p))
        
        # checking for engine speed limit (engine-, drag-, or cornering- limited)
        v = min(v, veh.v_max, v_drag)
        
        
        #We have solved for the initial velocity; now we must compensate to ensure ax_tot = 0 
        #and to ensure we stay on friction ellipse
        
        # adjust speed for drag force compensation
        adjust_speed = True
        
        while adjust_speed:
            # aero forces
            downforce = 0.5*veh.rho*veh.factor_Cl*veh.Cl*veh.A*v**2
            drag = 0.5*veh.rho*veh.factor_Cd*veh.Cd*veh.A*v**2
            
            # rolling resistance
            rolling_resistance = veh.Cr*(-downforce+Wz)
            
            # normal load on driven wheels
            Wd = (factor_drive*Wz+(-factor_aero*downforce))/driven_wheels
            
            # drag acceleration
            ax_drag = (drag+rolling_resistance+Wx)/M
            
            # maximum lateral acc available from tyres
            ay_max = np.sign(r)/M*(muy+dmy*(Ny-(Wz-downforce)/4))*(Wz-downforce)
            
            # needed lateral acc in order to make the turn
            ay_needed = r*v**2+g*np.sin(bank) # account for circular motion and track banking
            
            #All derived in notebook
            
            # calculating driver inputs
            if ax_drag <= 0: # need throttle to compensate for drag
                # max long acc available from tyres
                ax_tyre_max_acc = 1/M*(mux+dmx*(Nx-Wd))*Wd*driven_wheels #all driven wheel formulas are from OpenLap videos
                #will be re-derived in notes **************** TODO
                
                # getting power limit from motor
                #TODO again
                ax_power_limit = 1/M*np.interp(v, veh.vehicle_speed, veh.factor_power*veh.fx_engine)
                
                # available combined lat acc at ax_net==0 => ax_tyre==-ax_drag
                ay = ay_max*np.sqrt(1-(ax_drag/ax_tyre_max_acc)**2) # friction ellipse
                
                # available combined long acc at ay_needed

                ax_acc = ax_tyre_max_acc*np.sqrt(1-(ay_needed/ay_max)**2) # friction ellipse
                
                # getting tps value
                scale = min(-ax_drag, ax_acc)/ax_power_limit
                tps = max(min(1,scale), 0) ; # making sure its positive
                bps = 0 # setting brake pressure to 0
            
            else: # need brake to compensate for drag
                # max long acc available from tyres
                ax_tyre_max_dec = -1/M*(mux+dmx*(Nx-(Wz-downforce)/4))*(Wz-downforce)
                
                # available combined lateral acc at ax_net==0 => ax_tyre==-ax_drag
                ay = ay_max*np.sqrt(1-(ax_drag/ax_tyre_max_dec)**2) # friction ellipse
                
                # available combined long acc at ay_needed
                ax_dec = ax_tyre_max_dec*np.sqrt(1-(ay_needed/ay_max)**2) # friction ellipse
                
                # getting brake input
                fx_tyre = max(ax_drag, -ax_dec)*M #***TODO: check that this is positive
                bps = max(fx_tyre, 0)*veh.beta # making sure its positive
                tps = 0 # setting throttle to 0
            
            # checking if tyres can produce the available combined lateral acc
            if ay/ay_needed < 1: # not enough grip; tweak v and go back through the calculation
                v = np.sqrt((ay-g*np.sin(bank))/r)-0.001 # the (-1E-3 factor is there for convergence speed)
                
            
            else: # enough grip
                adjust_speed = False
            
    return v
        