'''
This will be the main file
'''
import numpy as np
import lap_utils
import scipy.io as spy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#Import track and vehicle files like OpenLap
import track as tr
import vehicle as veh

#Track and vehicle files are located in the respective python scripts


def simulate():
    # Maximum speed curve
    v_max = np.zeros(tr.n, dtype=np.float32)
    bps_v_max = np.zeros(tr.n, dtype=np.float32)
    tps_v_max = np.zeros(tr.n, dtype=np.float32)
    
    for i in range(tr.n):
        v_max[i], tps_v_max[i], bps_v_max[i] = lap_utils.vehicle_model_lat(veh, tr, i)
    
    # HUD
    print('Maximum speed calculated at all points.')
    
    # Finding apexes
    apex, _ = find_peaks(-v_max)  # Assuming findpeaks is a function that finds peaks in an array
    
    v_apex = v_max[apex]  # Flipping to get positive values
    

    # Setting up standing start for open track configuration
    if tr.info.config == 'Open':
        if apex[0] != 1:  # If index 1 is not already an apex
            apex = np.insert(apex, 0, 1)  # Inject index 1 as apex
            v_apex = np.insert(v_apex, 0, 0)  # Inject standing start
        else:  # Index 1 is already an apex
            v_apex[0] = 0  # Set standing start at index 1
    
    # Checking if no apexes found and adding one if needed
    if len(apex)==0:
        apex = np.argmin(v_max)
        v_apex = v_max[apex]

    
    
   
    
    
    # Reordering apexes for solver time optimization
    old_apexes = v_apex #for plotting later
    apex_table = np.column_stack((v_apex, apex))
    apex_table = apex_table[apex_table[:, 0].argsort()]
    
    v_apex = apex_table[:, 0]
    apex = apex_table[:, 1]
    
    apex = apex.astype(int)
    
    

    # Getting driver inputs at apexes
    tps_apex = tps_v_max[apex]
    bps_apex = bps_v_max[apex]
    
    
    
    # Memory preallocation
    N = len(apex)  # Number of apexes
    print(N)
    flag = np.zeros((tr.n, 2), dtype=bool)  # Flag for checking that speed has been correctly evaluated
    v = np.full((tr.n, N, 2), np.inf, dtype=np.float32)
    ax = np.zeros((tr.n, N, 2), dtype=np.float32)
    ay = np.zeros((tr.n, N, 2), dtype=np.float32)
    tps = np.zeros((tr.n, N), dtype=np.float32)
    bps = np.zeros((tr.n, N), dtype=np.float32)
    
    counter = 0
    skipped = 0
    
    # Running simulation
    for k in range(2):  # Mode number
        mode = 1 if k == 0 else -1
        k_rest = 1 if k == 0 else 0
    
        for i in range(N):  # Apex number
            if not (tr.info.config == 'Open' and mode == -1 and i == 0):  # Does not run in decel mode at standing start in open track
                # Getting other apex for later checking
                i_rest = lap_utils.other_points(i, N)
                # Getting apex index
                j = int(apex[i])
                print("Here j = {}".format(j))
                # Saving speed, latacc, and driver inputs from presolved apex
                v[j, i, k] = v_apex[i]
                
                
                ay[j, i, k] = v_apex[i] ** 2 * tr.r[j]
                tps[j, :] = tps_apex[i] * np.ones(N)
                bps[j, :] = bps_apex[i] * np.ones(N)
                # Setting apex flag
                flag[j, k] = True
                # Getting next point index
                _, j_next = lap_utils.next_point(j, tr.n-1, mode)
    
                if not (tr.info.config == 'Open' and mode == 1 and i == 0):  # If not in standing start
                    # Assuming the same speed right after apex
                    v[j_next, i, k] = v[j, i, k]
                    # Moving to the next point index
                    j_next, j = lap_utils.next_point(j, tr.n-1, mode)
    
                while True:
                    # Calculating speed, accelerations, and driver inputs from the vehicle model
                    v[j_next, i, k], ax[j, i, k], ay[j, i, k], tps[j, k], bps[j, k], overshoot = lap_utils.vehicle_model_comb(veh, tr, v[j, i, k], v_max[j_next], j, mode)
                    print("j = {} | | v = {}".format(j, v[j_next, i, k]))
                    # Checking for limit
                    if overshoot:
                        #print("Overshot")
                        break
                    # Checking if the point is already solved in the other apex iteration
                    if flag[j, k]:
                        if np.max(v[j_next, i, k] >= v[j_next, i_rest, k]) or np.max(v[j_next, i, k] > v[j_next, i_rest, k_rest]):
                            skipped += 1
                            break
                    # Updating flag and progress bar
                    flag[j, k] = True #flag_update(flag, j, k, prg_size, logid, prg_pos)
                    # Moving to the next point index
                    j_next, j = lap_utils.next_point(j, tr.n-1, mode )
                    # Checking if the lap is completed

                    if tr.info.config == 'Closed':
                        if j == apex[i]:  # Made it to the same apex
                            counter += 1
                            break
                    elif tr.info.config == 'Open':
                        if j == tr.n:  # Made it to the end
                            flag[j, k] = True #flag_update(flag, j, k, prg_size, logid, prg_pos)
                            break
                        if j == 1:  # Made it to the start
                            break

    print("Counter: {}".format(counter))
    print("Skipped: {}".format(skipped))

    #print(v[:, 20, 0])
    # preallocation for results
    V = np.zeros(tr.n)
    AX = np.zeros(tr.n)
    AY = np.zeros(tr.n)
    TPS = np.zeros(tr.n)
    BPS = np.zeros(tr.n)
    
    # solution selection
    for i in range(tr.n):
        IDX = v.shape[1]
        #min_values = np.min([v[i, :, 0], v[i, :, 1]], axis=0)
        #idx = np.argmin(min_values)
        #V[i] = min_values[idx]
        min_values = np.min(v[i, :, :], axis=1)
        idx = np.argmin(min_values)
    
        V[i] = min_values[idx]
    
        if idx <= IDX:  # solved in acceleration
            AX[i] = ax[i, idx, 0]
            AY[i] = ay[i, idx, 0]
            TPS[i] = tps[i, 0]
            BPS[i] = bps[i, 0]
        else:  # solved in deceleration
            AX[i] = ax[i, idx - IDX, 1]
            AY[i] = ay[i, idx - IDX, 1]
            TPS[i] = tps[i, 1]
            BPS[i] = bps[i, 1]
        
            
    
    #print(flag)
    #Below is a useful check to see where the apexes are:
    x = []
    xapex = []
    
    for i in range(len(v_max)):
        x.append(i)
        
        if i in apex:
            xapex.append(i)

    ax = plt.axes()
    ax.plot(x, v_max)
    ax.plot(x, tr.r*50)
    ax.scatter(xapex, old_apexes)
    
    #print(V)

    ax.plot(x, V, color='r')
    plt.show()
    # laptime calculation    
    dt = np.divide(tr.dx, V)
    time = np.cumsum(dt)
    
    #max_sector = int(np.max(tr.sector))
    #sector_time = np.zeros(max_sector)
    
    #for i in range(1, max_sector + 1):
        #sector_time[i - 1] = np.max(time[tr.sector == i]) - np.min(time[tr.sector == i])
    
    laptime = time[-1]
    
    print("Laptime is {}".format(laptime))




def test():
    p = 30
    v_max, tps_v_max, bps_v_max = lap_utils.vehicle_model_lat(veh, tr, p)
    print(v_max)

simulate()


def simulate2():
    #Maximum speed trace
    v_max = []
    v_apex = []
    
    for i in range(tr.n):
        v_max.append(lap_utils.vehicle_model_lat(veh,tr,i))
    
    
    #Find local minima of v_max (maximum speed curve)
    
    apexes, _ = find_peaks(np.reciprocal(v_max)) #findpeaks finds the maxima, so we invert first (it has some issues with negative)
    
    for apex in apexes:
        v_apex.append(v_max[apex]) #the local minima
    
    N = len(apexes)
    
    
    '''
    Main Simulation Loop:
    
        For each apex:
            Accelerate the vehicle to every other mesh point forward in time
            Decelerate vehicle to every other mesh point backward in time
            
            This will give two segments/arrays (accel and decel) for each apex point
        
        The final velocity solution is the minimum velocity of all of these segments
    '''
    
    #Accelerate and Decelerate from each Apex to find the v at each mesh point:
    
    #1st dimension is the mesh point index
    #2nd dimension is the apex index (which apex we solved this set of velocities for)
    #3rd dimension indicates acceleration or deceleration
    
    v = np.inf*np.ones(shape=(tr.n, N, 2)) #we set to a large number because final v is the minimum of value of each segment
    
    ax = np.zeros(shape=(tr.n, N, 2))
    ay = np.zeros(shape=(tr.n, N, 2))
    tps = np.zeros(shape=(tr.n, N, 2)) #throttle pressure -> talk with e-powertrain; this is how we calculate Energy consumption
    bps = np.zeros(shape=(tr.n, N, 2))
    
    flag = np.zeros((tr.n,2)) # optimization flag; if true, speed has been correctly evaluated for that mesh point
    
    for k in range(2):
        mode = 2*k-1 #mode is -1 for decel or 1 for acceleration
        
        for i in range(N): #for each apex
            j = apexes[i]
            
            v[j][i][k] = v_apex[i]            #initial velocity is just the ith apex velocity 
            #tps(j,:) = tps_apex(i)*ones(1,N) ;
            #bps(j,:) = bps_apex(i)*ones(1,N) ;
            flag[j][k] = 1 #1 = True, 0 = false
    
            j_next, j = lap_utils.next_point(j,tr.n-1,mode)
            
            
            while True: 
                
                v[j_next,i,k], ax[j,i,k], ay[j,i,k], tps[j,k], bps[j,k], overshoot = lap_utils.vehicle_model_comb(veh,tr,v[j,i,k],v_max[j_next],j,mode)
                
                # if we have solved this point in previous apex iterations
                if (flag[j,k] == 1):
                    # and if ANY of the previous solutions yield a smaller velocity, 
                    vj = [] #other solutions at jmax
                    for h in range(2):
                        for l in range(N):
                            if not(h==k and l==i):
                                vj.append(v[j_next,l,h])
                                
                    
                    if (v[j_next,i,k] > np.max(vj)):
                        break # we don't need to keep solving this direction (i.e. accel or decel) from this apex
        
                # updating flag; we have solved for this point
                flag[j][k] = 1
                
                # moving to next point index (increments j and j_next, mod tr.n)
                j_next, j = lap_utils.next_point(j,tr.n-1,mode)
    
                if j==apexes[i]: #repeat until we get back to the initial apex point
                    break
        
        
        
        '''
        Purpose of flag:
            Accelerate and decelerate from each apex point i to all other mesh points
            Solve for each mesh point j
                If we have already solved for (j,k) and found a smaller maximum speed, 
                it's not worth it to solve for other mesh points from this apex point, as 
                we can be sure that this will be the case for all other mesh points as calculated from i
                So we skip to the next apex point
    
        '''
        
    
    
    print("here")
    print(v[0, :, 0])
    # preallocation for results
    V = np.zeros(tr.n)
    AX = np.zeros(tr.n)
    AY = np.zeros(tr.n)
    TPS = np.zeros(tr.n)
    BPS = np.zeros(tr.n)
    
    # solution selection
    for i in range(tr.n):
        IDX = v.shape[1]
        #min_values = np.min([v[i, :, 0], v[i, :, 1]], axis=0)
        #idx = np.argmin(min_values)
        #V[i] = min_values[idx]
        min_values = np.min(v[i, :, :], axis=1)
        idx = np.argmin(min_values)
    
        V[i] = min_values[idx]
    
        if idx <= IDX:  # solved in acceleration
            AX[i] = ax[i, idx, 0]
            AY[i] = ay[i, idx, 0]
            TPS[i] = tps[i, 0]
            BPS[i] = bps[i, 0]
        else:  # solved in deceleration
            AX[i] = ax[i, idx - IDX, 1]
            AY[i] = ay[i, idx - IDX, 1]
            TPS[i] = tps[i, 1]
            BPS[i] = bps[i, 1]
        
        
        




    # laptime calculation    
    dt = np.divide(tr.dx, V)
    time = np.cumsum(V)
    
    max_sector = int(np.max(tr.sector))
    sector_time = np.zeros(max_sector)
    
    for i in range(1, max_sector + 1):
        sector_time[i - 1] = np.max(time[tr.sector == i]) - np.min(time[tr.sector == i])
    
    laptime = time[-1]
    
    print("Laptime is {}".format(laptime))
