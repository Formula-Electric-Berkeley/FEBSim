import numpy as np
#Define state; vx, vy are in vehicle-oriented coordinates
vx = [] #longitudinal velocity
vy = [] #lateral velocity
psi_dot = [] #yaw angle rate of change
delta = [] #steering angle


#Define earth-frame coordinates (tracks real X,Y of the vehicle, with origin as a fixed point on the track)
Vx = []
Vy = []

#Define simulation parameters
dt = 1 #second
lf = 1 #length from COG to front axle in meters
lr = 1 #length from COG to rear axle in meters (length * weight distribution)
lb = lf + lr # length of vehicle
m = 1 #kg
Iz = 1 #moment of inertia about COM in z axis

#Primary sources: Auto Reference, Bicycle Model Path Planning Theory PDFs on the Bicycle Model Notion


#Objective is to solve for curves using bicycle model dynamics
def curve_solver(R):
    delta = find_delta(R)

    #Keeping with our circular arc assumption, lateral acceleration must still fit v^2 / R
    ay_needed = vx**2 / R

    #BCs: on a straight, delta, psi_dot = 0 


    Rx = 0
    Aero_drag = 0
    ax_drag = (Aero_drag + Rx) / m
    #
                
    
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


def curve():
    '''
    Make an initial guess for delta between two points
    Move forward by a point, track error in the final position 
    Move forward until the distance travelled = the arc length
    
    We know how the vehicle will progress in time given steering angle and tire forces
    Longitudinal acceleration is limited by (a) tire grip, (b) drag forces, or (c) throttle
    

    Crucial point: x and y axes are aligned with the *vehicle* not the track like we did previously

    X and Y axes are aligned with the earth, absolute coordinates
    Vx = vx * cos(psi) - vy * sin(psi)
    Vy = vx * sin(psi) + vy * cos(psi)


    xt and yt axes are aligned with the track
    ^ we can write vxt and vyt in terms of vx and vy

    If the car is perfectly following the racing line, vx should be along the line
    Racing line traces motion of COG
    This means we *do* have constraints on vx and vy



Fy = Fy_tyres - drag
ay = Fy/m

    General procedure:

    Find the necessary steering angle
    - assume steady-state initially; compute from update dynamics 

    Guess an initial velocity v_long (bounded like RoseLap does)
    determine the required lateral acceleration (a = v_long^2 / r)
    ay_needed = ...

    Use this to determine the lateral tyre forces by rearranging our equations
    - assume initial state of the vehicle is known
    - assume continuous angular velocity at the apex of the curve (psi_dot_dot = 0 here)
    - assume pure lateral (ax = 0) at apex
    - tire circle to determine Fxr given Fyr (where do we get this?)
    - solve 4 equations for 4 unknowns
    
    Make sure we don't fly off
    - if we do, iterate on v_long until we get the right guess

    Calculate how much grip we have left from the tires
    - How do we do that when the equations are coupled?
    - We have at most 3 equations at 4 unknowns, unless we assume we know Fxr or group together Fxf and Fyf

    Determine how much longitudinal grip is needed to counteract frictional forces



    Meeting pseudocode:
    1. travel across the curve once and get steering angle
    2. determine maximum tire forces based on steering angle
        at the v_apex, we want pure lateral acceleration
        ay_needed = v^2 / r

    3. 

    
    
    Steering angle pseudocode:
    1. Initial guess of steering angle (/ tangent of curve)
    2. Progress vehicle in time using update dynamics
    3. Progress vehicle until the total distance travelled = dx between points
    4. Compare position error and adjust steering angle

    Update dynamics pseudocode:
    1. Given current velocity and yaw angle of the car, determine the current slip angles
    2. Pull the corresponding tire circle for this slip angle from tire data
        -> given Fxf, we constrain Fyf and vice versa
    3. Calculate the lateral acceleration needed to maintain the circular motion (ay = v^2 / r)
    4. Initial speed solution 
        - straight case: from engine-limitation and drag-limitation
        - cornering case: also must solve the differential equation (like openlap)
        - we don't have a clear diff. eq, so just try to solve pure lateral acceleration (ax = 0)
        - also need to use ay = V^2 / r
        - what are the tire forces? attempt to maximize ay, V given ; very circular logic. We need v to get slip angles, 
        so we must have some kind of transience

    Mesh connector: This is what update dynamics is for
    1. Given the current state, accelerate or decelerate backwards as fast as possible to the other mesh nodes
    2. Given the v_max at the next mesh node, determine ax_max; ax_track-limited = ax_max-ax_drag
    3. Determine the maximum lateral acceleration 
    2. We know the tire circle and steering angle. 


    2/29/24 overhaul

    Vehicle lateral solver
    1. Guess steering
    2. Determine alpha from steering and state of vehicle
    3. Determine maximum lateral tire force
        - Flong = F_drag so net ax = 0
    4. Guess velocity using Flat/m = v^2 / R
    5. Maximum lateral force can be estimated by mv^2 / r
    6. Use tire force slip angle data to determine the slip angle
    7. Determine steering angle from slip angles
     
    Lateral solver
    1. Assume pure lateral F_lat_max to initial guess v
    2. At this velocity, determine Fdrag
    3. Adjust v guess:
        - Fthrottle + Fdrag = 0 constrains Fx
        - Return to tire circle (assume no slip) to get Fy
    4. But if we fix slip angle, how can we get steering?

    -> ask chatgpt
    self-aligning tire moments
    psi_dot = v/l (tan(delta) - Lr/L *alpha_r + Lf/L *alpha_f) = 0

    Iterate over possible steering angles and slip angles to find the combination that satisfies the ideal lateral force balance

    Guido's steering
    1. Take Flat = mv^2 / R
    2. Determine slip angles from F
    3. Use geometry to determine approximate steering angle


    V_lateral_max
    1. Assume pure lateral (mv^2 / R)
    2. We need some F to coutneract drag, sample tire circle
    3. we need to know the slip angle to know what circle to use
        - Slip angles can be calculated from the current state of the vehicle
        - assuming we know the initial state, we just propagate forward transiently using "update dynamics" 
            to get the current state
    4. Adjust Fs accordingly
    5. Use Fs and slip angles to determine a guess at steering angle

    6. Validate by using the update_dynamics code in its current form and checking if we overshoot


    Note* They have longitudinal as a function of slip ratio, not alpha

    
    Guido Big time
    1. Determine Fy max using friction coefficient and normal force on the tire
        - remember that we need to consider each wheel
    2. Fy_max = mv^2 / R
    3. Solve for v using Fy_max = mu*Fz
    

    When considering dependency on slip angle
    1. Determine slip angles based on steering angle and current vehicle state
    2. Apply the tire model to determine the forces based on the slip angles
    3. Solve the equations of motion (iterative solutions or numerical methods)
    4. Optimize this for maximum lateral force



    v_initial = np.sqrt(mu * g * R)

    for _ in range(100):  # Limit iterations to prevent infinite loops
        # Calculate slip angles for front and rear tires assuming small angle approximation
        delta_optimal = np.arctan((2 * Lr) / (Lf + Lr))  # Optimal slip angle for uniform load distribution
        vy = v_initial / R  # Lateral velocity for centripetal acceleration
        r = v_initial / R  # Yaw rate for circular motion
        
        alpha_f = delta_optimal - np.arctan((vy + Lf * r) / v_initial)
        alpha_r = -np.arctan((vy - Lr * r) / v_initial)

        # Calculate lateral forces using the linear tire model
        Fyf = Cf * alpha_f
        Fyr = Cr * alpha_r

        # Calculate the resultant lateral force
        Fy_total = Fyf + Fyr

        # Calculate the maximum lateral acceleration based on the tire forces
        ay_max = Fy_total / mass

        # Update the maximum velocity estimate based on the calculated lateral acceleration
        v_new = np.sqrt(ay_max * R)

        # Check for convergence (i.e., the change in estimated velocity is small)
        if np.abs(v_new - v_initial) < 1e-3:
            break

        v_initial = v_new

    
    General plan:
    1. Initial guess of v assuming maximum Fy
        - (Assume we know steering angle)
    2. Run update_dynamics to get alphas
    3. Use alphas to get maximum lateral forces
        - Consider tire circle to get Fx = Fdrag
    4. Use max lateral forces to determine ay_max
    5. Use ay_max to get v
    6. Repeat, hope it converges

    Q. is there a better initial guess for v? 
    Q. The steering angle shown does not depend on the radius of curvature. 
    Should we iterate to find the optimal steering angle?

    Just raw iterate over delta to find the optimal speed
     - when doing this, make sure we don't fly off the track (ay = v^2 / R)
     - Assuming we have the real F max, we can just check that v is below sqrt(R*ay)

     

    Semi-transient model:

    # Calculate accelerations
    for _ in range(depth_of_sim):
        # Calculate forces
        Fxf = 0  # Placeholder for front tire force
        Fxr = 0  # Placeholder for rear tire force
        Rx = 0  # Placeholder for rolling resistance force
        F_drag = 0  # Placeholder for aerodynamic drag force
        F_throttle = 0  # Placeholder for throttle force

        ax = vy * psi_dot + (Fxf * np.cos(delta) - Cf * np.sin(delta) + Fxr - Rx - F_drag + F_throttle) / mass
        ay = -vx * psi_dot + (Cf * np.cos(delta) + Fxf * np.sin(delta) + Cr) / mass
        psi_dot_dot = (Cr * lr + lf * (Fxf * np.sin(delta) + Cf * np.cos(delta))) / Iz

        # Update state variables
        vx += ax * dt
        vy += ay * dt
        psi_dot += psi_dot_dot * dt
        -> Chat 3 wants us to progress the sim forward in time 
        -> Eventually, after enough time traversing a circular arc, we should approach the steady-state solution
        -> initial guess at vx, vy, psi_dot; then trace the curve using the Fxfs and Fxrs they give
        -> Fxf, Fxr max are dependent on slip angle
        -> how do we know what to maximize? We want to maximize Fy_total while ensuring Fx is sufficient -> v_new
        -> sample various V, delta until we get the best v_new

        When we sample V, what are we actually changing?

        # Calculate lateral forces using the linear tire model
        alpha_f = np.arctan((vy + lf * psi_dot) / vx)
        alpha_r = -np.arctan((vy - lr * psi_dot) / vx)
        Fyf = Cf * alpha_f
        Fyr = Cr * alpha_r

        # Calculate the resultant lateral force
        Fy_total = Fyf + Fyr

        # Calculate the maximum lateral acceleration
        ay_max = Fy_total / mass

        # Update the maximum velocity estimate
        v_new = np.sqrt(ay_max * R)

        if v_new > v_max:
            v_max = v_new
            delta_optimal = delta

        # Break if the change in estimated velocity is small
        if np.abs(v_new - v_initial) < 1e-3:
            break

        v_initial = v_new

    return v_max, delta_optimal


    Slip ratio kappa = (omega*R/V - 1)

    omega = angular velocity of the tire spinning
    R = effective radius of the tire (we know as a function of vertical load...)
    V = forward velocity of the vehicle

    ^ known based on our state

    '''

#sim parameters
max_angle = 38 #max angle for steering in degrees
depth_of_sim = 100
angles_checked = 50 #number of steering angles to sample in the range
force_tuples_checked = 100
tolerance = 1e-2 #to check agreement between the heading tangents

def guido_time(num_nodes, tangents, curve_types):
    
    current_state = [0, 0, 0, 0] #vx, vy, psi_dot, delta 
    #assume resting start for now

    states = [current_state]
    deltas = [0]

    for node in range(num_nodes-1):
        #****solve the vehicle state at the next mesh point
        
        #optimization for steering angle selection (sample only the angles we need)
        curve_type = curve_types[node] 
        #Todo: narrow down delta_range to a narrower band around the kinematic prediction 

        if curve_type == 'Straight':
            #regular open_lap solution
            delta_range = [0]
        elif curve_type == 'Right':
            #possible steering angle range (switch signs depending on left or right)
            delta_range = np.linspace(0, max_angle, angles_checked) 
        elif curve_type == 'Left':
            delta_range = np.linspace(-max_angle, 0, angles_checked) 
            
        delta_range *= np.pi/180 #convert to radians

        tangent = tangents[node+1] #get the tangent of the racing line at the next point
        #the velocity at the next point should be parallel to the tangent of the racing line

        current_state, delta_optimal = get_next_state(current_state, delta_range, tangent)


        states.append(current_state)
        deltas.append(delta_optimal)




def get_next_state(current_state, delta_range, tangent):
    # optimal speed v_max and steering angle delta 
    v_max = 0
    delta_optimal = 0
    next_state = [0, 0, 0]
    vx, vy, psi_dot = current_state

    for delta in delta_range:
        #calculate slip angles based on current_state+delta
        alpha_f = delta - np.arctan((vy + lf * psi_dot) / vx)
        alpha_r = -np.arctan((vy - lr * psi_dot) / vx)

        #calculate tire ellipses from the alphas
        #returns an array of (Fx, Fy) along the tire_ellipse
        front_forces = get_ellipse(alpha_f)
        rear_forces = get_ellipse(alpha_r)

        #determine non-tire forces
        Rx = 0 #rolling resistance, a function of longitudinal velocity; negligible rolling resistance in the y
        F_drag = 0 #drag force in the longitudinal direction

        F_throttle = 0 #ensure the tire forces commanded do not exceed the max throttle force
        #is throttle included in tire forces? do better analysis of this later ***BIG TODO

        #drag force has some component in the psi_dot_dot direction

        #everything else is manifested in the tire forces; friction, rolling resistance, and throttle
        #the final force moving the car is the tire force
        #the throttle/braking force acts higher up stream and gives us the friction force
        #the net friction force in the wheels will come from
        # Fxf: (throttle - rolling resistance), braking
        # physically, how does throttle play a role? That should be added to ax
        # friction is what's moving us forward, but we cannot friction forward any greater than F_throttle - Rx

        #so, we know the maximum friction the wheel can give in the x
        #we also know the maximum throttle force as Fthrottle - Rx [crucial note--no sliding friction force!]
        #F_aero_drag also acts in - x -- even if it doesn't act through the tires, U - Rx - Drag > 0 will be max + force in x
        #Max - force in the x: U - Rx - Drag (U < 0 implies braking)

        #How to determine U?
        #Openlap needs to constrain ay
        #If we have ay = v^2 / R, this can constrain U!

        '''
        theta = angle of the track tangent
        a_perp = v^2 / R
        a_parallel = 0
        

        '''

        #Check:
        #go forward a timestep
        #first, make sure we're on the track
        #then make sure the velocity vector is // to the track

        #calculate the distance we travelled (interpolate our current and next-step velocity)
        #if we go |distance| along the track from our current position, sample the track at that point
        #calculate the tangent at that point and make sure our car's velocity is //

        


        for front_pair in front_forces:
            for rear_pair in rear_forces:
                Fxf, Fyf = front_pair
                Fxr, Fyr = rear_pair

        
                #Must project the tire-frame force to the vehicle frame (shown in Auto Reference)
                ax = vy*psi_dot + (Fxf*np.cos(delta) - Fyf*np.sin(delta) + Fxr - Rx - F_drag + F_throttle)/m
                ay = -vx*psi_dot + (Fyf*np.cos(delta) + Fxf*np.sin(delta) + Fyr)/m
                psi_dot_dot = (Fyr*lr + lf*(Fxf*np.sin(delta)+Fyf*np.cos(delta)))/Iz 
                
                #appears like Fyf is defined inwards, check hand-notes for sign conventions

                #Finite difference increment for well-behaved changes (along a curve of near-constant curvature)
                vx = vx + ax*dt
                vy = vy + ay*dt
                psi_dot = psi_dot + psi_dot_dot*dt
                

                #check that the velocity solution matches with the track
                #**optimization, move this check further upstream
                #do we need to include the yaw, or is good enough that the instantaneous velocity 
                #matches in direction with the tangent of the curve?
                #

                
                #check if this is the fastest valid solution
                v = vx**2 + vy**2

                if v > v_max:
                    v_max = v
                    delta_optimal = delta
                    next_state = [vx, vy, psi_dot]



    return (next_state, delta_optimal)



      

def get_ellipse(alpha):
    #returns N tuples of (Fx, Fy) sampled from the circumference of the tire ellipse
    forces = []
    return forces
   
def max_velocity_solver(R):
    delta_range = np.linspace(-38, 38, 50) #possible steering angle range
    delta_range *= np.pi/180 #convert to radians
    depth_of_sim = 100

    vx = np.sqrt(mu * g * R) #initial speed estimate

    # optimal speed v_max and steering angle delta 
    v_max = 0
    delta_optimal = 0

    for delta in delta_range:

        for _ in range(depth_of_sim):  # Velocity iteration limit
            vy = v_initial / R  # Lateral velocity for centripetal acceleration
            psi_dot = v_initial / R  # Yaw rate for circular motion

            # Calculate slip angles
            #alpha_f = delta - np.arctan((vy + Lf * psi_dot) / v_initial)
            #alpha_r = -np.arctan((vy - Lr * psi_dot) / v_initial)

            alpha_f = np.arctan((vy+lf*psi_dot)/vx) #intuitively, should depend on vx,vy; not Vx, Vy
            alpha_r = -np.arctan((vy-lr*psi_dot)/vx)

            # Calculate lateral forces using the linear tire model
            Fyf = Cf * alpha_f
            Fyr = Cr * alpha_r

            # Calculate the resultant lateral force
            Fy_total = Fyf + Fyr

            # Calculate the maximum lateral acceleration
            ay_max = Fy_total / mass

            # Update the maximum velocity estimate
            v_new = np.sqrt(ay_max * R)

            if v_new > v_max:
                v_max = v_new
                delta_optimal = delta

            # Break if the change in estimated velocity is small
            if np.abs(v_new - v_initial) < 1e-3:
                break

            v_initial = v_new

    return v_max, delta_optimal



#Solves for vehicle state at the next time step, given steering angle delta and the throttle force F_throttle 
def update_dynamics(vx, vy, psi_dot, delta, F_throttle):
    #Determine the sideslip angles of front and rear tires 
    alpha_f = np.arctan((vy+lf*psi_dot)/vx) #intuitively, should depend on vx,vy; not Vx, Vy
    alpha_r = -np.arctan((vy-lr*psi_dot)/vx)
        # - does not explicitly depend on steering angle, but rather the velocities 

    Fxf, Fyf = max_tyre_forces(alpha_f) #calculates "longitudinal" tire force *in the frame of the tire*
    Fxr, Fyr = max_tyre_forces(alpha_r)

    Rx = 0 #rolling resistance, a function of longitudinal velocity; negligible rolling resistance in the y
    F_drag = 0 #drag force in the longitudinal direction

    #Must project the tire-frame force to the vehicle frame (shown in Auto Reference)
    ax = vy*psi_dot + (Fxf*np.cos(delta) - Fyf*np.sin(delta) + Fxr - Rx - F_drag + F_throttle)/m
    # Not exactly true; strictly, the tire forces should impose a maximum on throttle/breaking (and rolling resistance should be included in this)

    ay = -vx*psi_dot + (Fyf*np.cos(delta) + Fxf*np.sin(delta) + Fyr)/m

    psi_dot_dot = (Fyr*lr + lf*(Fxf*np.sin(delta)+Fyf*np.cos(delta)))/Iz #appears like Fyf is defined inwards, check hand-notes for sign conventions

    #Finite difference increment for well-behaved changes (along a curve of near-constant curvature)
    vx = vx + ax*dt
    vy = vy + ay*dt
    psi_dot = psi_dot + psi_dot_dot*dt

    return vx, vy, psi_dot


    


def find_delta(endpoint1, endpoint2, R):
    #Initial guess based on kinematic bicycle model 
    La = np.sqrt((endpoint2[0]-endpoint1[0])**2 + (endpoint2[1]-endpoint1[1])**2) #stright line distance between two points
    delta = np.arctan(2*lb*np.sin(2*np.arcsin(La/(2*R)))/La)

    THRESHOLD = 0.1 #0.01 in delta is about 0.6 degrees
    #Assume initial guess is good enough for now

    return delta
    


#Assume we know exactly what Fx we need to stay on the tire circle with a given Fy
def tyre_circle(alpha): #calculate the full circle at slip angle alpha
    Fx = 1
    Fy = 1
    return (Fx, Fy)



#Determines lateral and longitudinal forces on a tire given the slip angle alpha
#When ready, should sample our Pacejka function
#We will likely need to double the tire forces (grouping front two and back two wheels together)
def max_tyre_forces(alpha):
    Fx = 10*alpha
    Fy = 10*alpha #for now, just assume constant of proportionality = 10
    return (Fx, Fy)