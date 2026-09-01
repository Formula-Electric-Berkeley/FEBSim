#Importing Modules

import numpy as np 
import matplotlib.pyplot as plt
import sys

#values to pull from mass point sim or real data 
v_cornner=30 #m/s
#checked out the tire data will def use for cornnering sim for temp after finish braking 

#Defining Physical Constants
T0=293.0 #ambient temp
G=9.81 #m/s^2
M=300 #kg mass of car assuming 
C=.01 #Coefficient of Rolling Resitance Std value
r_tire=.3 #m tire radius
w=.2 #m tire estimated width
rho_tire=1270 #kg/m^3 avg value of tire rubber density
rho_rim=2640 #kg^m^3 aluminum assumption
c_tire=1500 #J/kg K heat capcity of tire rubber 
c_rim=938 #J/kg K heat capcity
alpha_tire=.1 *10**(-6) #thermal diffusivity of tire -> thermal diffusivity alpha= k/rho*c
alpha_rim= 6 *10**(-5) #thermal diffusivity of rim 

#clumped model heat anaylsis from brake pad to hub
#units [=] W/mK
R_pad= 3.0
R_disc=.03
R_hub=.06

#for clumped modeling heat cond find fraction makes to wheel via assuming air, brake pad, and hub have intial temp do Qdot_hub/Qdot_tot Qdpt=DT/R
frac=(1/R_hub)/((1/R_hub)+(1/R_disc)+(1/R_pad))

#Difrentials/ Mesh Sizing
dr= 0.005 #m
dt=.1 #s
dtheta=np.deg2rad(20.0)

#Meshing 

r=np.arange(0,r_tire,dr)

theta=np.arange(0,2*np.pi,dtheta)

#create mesh grid-> pairs of r and theta temp and x and y map to
R,THETA=np.meshgrid(r,theta)

MESH=np.zeros(R.shape) #generating as mesh wher i index is R and j inex is THETA

#Material property assignment to mesh

alpha=np.zeros(np.shape(MESH))

temp=np.full(np.shape(MESH),T0)

rho=np.zeros(np.shape(MESH))

c=np.zeros(np.shape(MESH))

q=np.zeros(np.shape(MESH))

for i in range(len(r)):
     if r[i]<=.01:
        alpha[:,i]=alpha_rim
        c[:,i]=c_rim
        rho[:,i]=rho_rim
     else:
         alpha[:,i]=alpha_tire
         c[:,i]=c_tire
         rho[:,i]=rho_tire

#time for sim
Time_Limit=100 #sec
Time=0 # sec

#synthetic data for sim 
a_accel=9.81 #generic acceleration m/s^2
a_decel=-9.81
a=a_accel

v=np.zeros(int(Time_Limit/dt))

#Defining Governing Equations

F_RR= lambda C,M,G: C*M*G #Restive Rolling force -> force that from the polymers resting deformation from energy loss of not fully decompressing polymer chains

#Heat calculations
r_heat = r[-2]
qv = lambda v:(F_RR(C,M,G)*v) / (20*w*r_heat*dtheta*dr) #calculation for volumetric heat gen for rolling resitance or the polymers above tire-ground interface unable to release energy -> estimate 3 points share this heat gen

#2D fintie difference from cylindrical heat equaiton-> returns next steps temperature
def heat_step(T, alpha, dt, r, dr, dtheta, TiP, TiM, TjP, TjM,rho,cp,qv):

    d2T_dr2 = (TiP - 2*T + TiM) / dr**2
    dT_dr   = (TiP - TiM) / (2*dr)
    d2T_dth2 = (TjP - 2*T + TjM) / dtheta**2

    return T + dt * (alpha*(d2T_dr2 + (1/r)*dT_dr + (1/r**2)*d2T_dth2) + qv/(rho*cp))

#r=0/center node special case heat equation
heat_step0= lambda T,alpha,dt,dr,TiP: T+4*alpha*dt*((TiP-T)/(dr**2))

### Heat Transfer sim ###

#setting heat gen spots / for rolling put 1 dr behind max radius and start at theta 270 degrees -> rotate clockwise
q_gen=np.zeros(np.shape(MESH))

rad1=268
rad2=rad1+20

#stability check before sim
for l in range(1, len(r)):   # skip r=0
    crit_rim  = alpha_rim*dt*(1/dr**2 + 1/(r[l]**2 * dtheta**2))
    crit_tire = alpha_tire*dt*(1/dr**2 + 1/(r[l]**2 * dtheta**2))
    if max(crit_rim, crit_tire) > 0.5:
        print("Unstable mesh near r =", r[l])
        ans=input('proceed (y/n)? ')
        if ans =='n':
            sys.exit(0)
        else:
            break

for k in range(0,int(Time_Limit/dt)):
    #velocity of synthetic acceleration
    if (k+1)<len(v):
        if k>=int(Time_Limit/dt)*.5:
            a=a_decel
        else:
            a=a_accel
        v[k+1]=v[k]+a*dt

    #for rotating heat gen compression
    omega=v[k]/r[-2]
    rad= omega*dt
    index_to_rotate=int(np.round(rad/dtheta))
    index_to_rotate=index_to_rotate%len(theta)
    
    #setting new heat gen spot
    q_gen[:,:]=0

    if a>0:
        if rad2 <= len(theta): #if within the 0-360 degrees rad2 aka no wrap around normal setting 
            q_gen[rad1:rad2,-5:-1]=qv(v[k])
            print(qv(v[k]))
        else: #for the case where rad1 is like 358 degrees so start there fill in to 360 and do the wrap around starting at 0 to wherever rad 2 ends
            q_gen[rad1:,-5:-1]=qv(v[k])
            q_gen[:rad2-len(theta),-5:-1]=qv(v[k])
            print(qv(v[k]))
    else: #put brake pad between 45 degrees 2 dr away from center and use delta KE for heat energy 90% going to brake pad assuming even energy distribution to wheels 
            # total vehicle KE drop over this step
             dKE_total = max(0.0, 0.5 * M * (v[k]**2 - v[k+1]**2)) if (k+1) < len(v) else 0.0

            # braking power assigned to one wheel
             Qdot_wheel = 0.25 * dKE_total / dt

            # split braking power by mechanism
             Qdot_hub_in = 0.9 * frac * Qdot_wheel     # conducted into inner rim/hub mesh region
             Qdot_tread_slip = 0.1 * Qdot_wheel        # tread slip/friction heating

            #  inner hub/rim deposit region
             hub_i1, hub_i2 = 1, 3          # avoid i=0
             hub_j1, hub_j2 = 0, len(theta) # all theta

             V_hub = 0.0
             for jh in range(hub_j1, hub_j2):
                for ih in range(hub_i1, hub_i2):
                    V_hub += r[ih] * dr * dtheta * w

             q_hub = Qdot_hub_in / V_hub if V_hub > 0 else 0.0
             q_gen[hub_j1:hub_j2, hub_i1:hub_i2] = q_hub

            #  tread slip deposit region
            # put it near outer radius and near contact patch
             q_gen[270:290,-2:-5]=Qdot_tread_slip/(r_heat*dtheta*w*60*dr)

    rad1=index_to_rotate
    rad2=rad1+3

    temp_old = temp.copy() #holder for last cycle temp
    temp_new = temp_old.copy() #creating the holder for calculated temps

    for j in range(len(theta)):
        for i in range(len(r)):
    
            jp = (j + 1) % len(theta) #so that way it wraps to zero when hit 360
            jm = (j - 1) % len(theta) #so when at j=0 wraps to 359--> for a%b learned python always returns a number between 0 and b-1-> asks what number could i add to a multiple to 360 to get -1 

            if i == 0:
                # center node
                temp_new[j,i] = heat_step0(
                    temp_old[j,i],
                    alpha[j,i],
                    dt,
                    dr,
                    temp_old[j,i+1]
                )

            elif i == len(r)-1:
                # outer boundary (placeholder)--> dichlret boundry conditions
                temp_new[j,i] = temp_old[j,i]

            else:
                # interior nodes
                temp_new[j,i] = heat_step(
                    temp_old[j,i],
                    alpha[j,i],
                    dt,
                    r[i],
                    dr,
                    dtheta,
                    temp_old[j,i+1],
                    temp_old[j,i-1],
                    temp_old[jp,i],
                    temp_old[jm,i],
                    rho[j,i],
                    c[j,i],
                    q_gen[j,i]
                )

    temp = temp_new


#make cartesian
theta_plot = np.append(theta, 2*np.pi) #didnt include 2pi as unique point since = 0 so adding back to complete circle of tire 
temp_plot = np.vstack([temp, temp[0:1, :]])

R_plot, THETA_plot = np.meshgrid(r, theta_plot)

X = R_plot*np.cos(THETA_plot)
Y = R_plot*np.sin(THETA_plot)

# contour plot
plt.figure(figsize=(6,6))
levels = np.linspace(temp_plot.min(), temp_plot.max(), 50)
cont = plt.contourf(X, Y, temp_plot, levels=levels)


plt.colorbar(cont, label="Value")
plt.gca().set_aspect('equal') # make plot fit window
plt.xlabel("x")
plt.ylabel("y")
plt.title("Tire Temperature")

plt.show()

print(np.isnan(temp).any(), np.isinf(temp).any()) #sanity check 