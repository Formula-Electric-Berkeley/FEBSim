"""
================
    READ ME
================
To execute the function to display pack stats, use this format:
    >>> pack_name = run_stats(series_count, parallel_count, segment_count)
For example, to run the stats of a pack with 16s4p segment and 7 segment count, run the file with the line below added at the end.
    >>> p16_4_7 = run_stats(16, 4, 7)
After running the file, the stats will be displayed in terminal.

The program displays the stats for the pack with input configuration, and gives warnings (if any) of anything exceeding the FSAE rules.

The bottom of the file has a function that runs through combinations of configuration with a given range and outputs valid configurations that didnt raise a warning.
"""
import discharge_curves
from scipy.interpolate import interp1d

class Cell_21700:
    def __init__(self):
        self.cell_data = {
            "name": "21700", 
            "weight": 70,
            "length": 70.15,
            "min_v": 2.5,
            "nom_v": 3.6,
            "max_v": 4.2,
            "DCIR": 0.016,
            "capacity": 15.5
            }
        '''
        **BASE UNITS
        weight (g)
        length (mm)
        min_v (V)
        nom_v (V)
        max_v (V)
        DCIR (Ohm)
        capacity (Wh)
        ''' 

class Pack(Cell_21700):
    def __init__(self):
        super().__init__()
        self.red_flag = 0
    def pack(self, series, parallel, segment):

        pack_series = series * segment
        limit_pack_v = 600 #V
        limit_segment_v = 120 #V
        limit_power = 80 #kW
        limit_i_draw_percell = 45 #A
        limit_capacity = 10000 #Wh, capacity higher than 10000Wh is unnecessary.
        limit_length = 900 #mm, Pack will be too long beyond 900mm.

        #Input argument datatype check
        try:
            parallel = int(parallel)
            series = int(series)
            segment = int(segment)
        except:
            raise ValueError("Parallel or Series count must be integers.")

        #Update stats for pack   
        self.pack_series = pack_series
        self.parallel = parallel
        self.series = series    
        self.cell_data["name"] = f"{pack_series}s{parallel}p_Pack"
        self.cell_data["#_of_cells"] = parallel * pack_series
        self.cell_data["weight"] = round(self.cell_data["weight"] * (parallel * pack_series), 3)
        self.cell_data["length"] = round(self.cell_data["length"] * segment, 3)
        self.cell_data["min_v"] = round(self.cell_data["min_v"] * pack_series, 3)
        self.cell_data["nom_v"] = round(self.cell_data["nom_v"] * pack_series, 3)
        self.cell_data["max_v"] = round(self.cell_data["max_v"] * pack_series, 3)
        self.cell_data["DCIR"] = round(parallel / (self.cell_data["DCIR"] * pack_series), 3)
        self.cell_data["capacity"] = round(self.cell_data["capacity"] * parallel  * pack_series, 3)
        self.cell_data["peak_i_draw_pack"] = round(80000 / self.cell_data["max_v"], 3)
        self.cell_data["peak_i_draw_percell"] = round(80000 / self.cell_data["max_v"] / parallel, 3)
        self.cell_data["nom_i_draw_pack"] = round(80000 / self.cell_data["nom_v"], 3)
        self.cell_data["nom_i_draw_percell"] = round(80000 / self.cell_data["nom_v"] / parallel, 3)
        self.cell_data["power"] = round(self.cell_data["peak_i_draw_pack"] * self.cell_data["max_v"])
        
        # what is our voltage currently at?
        self.cell_data["voltage"] = self.cell_data["max_v"]  

        # what is the total charge *removed* from the accumulator? in Wh       
        self.cell_data["discharge"] = 0
        self.discharge_polynomials = discharge_curves.return_polynomials()

        self.drain_error = False # have we over-drained this pack?
        
        #Print Results (with formatting)
        '''
        print("====================\n     FSAE LIMIT\n====================")
        print(f"Pack Voltage: {limit_pack_v}V\nSegment Voltage: {limit_segment_v}V\nMax Power: {limit_power}kW")
        print("====================\n CURRENT PACK STAT\n====================")
        print(f'Pack configuration: {self.cell_data["name"]}, {[series, parallel, segment]}\n\
Total # of Cells:                   {self.cell_data["#_of_cells"]} Cells\n\
Weight (Cell only):                 {self.cell_data["weight"] / 1000} kg\n\
Minimum Pack Length:                {round(self.cell_data["length"] * 1.1, 1)} mm\n\
Minimum Voltage:                    {self.cell_data["min_v"]} V\n\
Nominal Voltage:                    {self.cell_data["nom_v"]} V\n\
Peak Voltage:                       {self.cell_data["max_v"]} V\n\
Internal Resistance:                {self.cell_data["DCIR"]} Ohm\n\
Capacity:                           {self.cell_data["capacity"]} Wh\n\
Peak Pack Current Draw :            {self.cell_data["peak_i_draw_pack"]} A\n\
Peak Cell Current Draw:             {self.cell_data["peak_i_draw_percell"]} A\n\
Nominal Pack Current Draw :         {self.cell_data["nom_i_draw_pack"]} A\n\
Nominal Cell Current Draw:          {self.cell_data["nom_i_draw_percell"]} A\n\
Current Draw at 10% pack (80kW):    {round(80000 / (self.cell_data["min_v"]), 2)} A\n\
Current Draw at 10% pack (20kW):    {round(20000 / (self.cell_data["min_v"]), 2)} A\n\
Current Draw at 10% pack (10kW):    {round(10000 / (self.cell_data["min_v"]), 2)} A\n\
Pack Power:                         {round(self.cell_data["power"] / 1000)} kW')
        
        #Rule Compliance Checks
        print("====================\n      WARNINGS\n====================")
        if self.cell_data["max_v"] > limit_pack_v:
            print(f"**Pack Peak Voltage greater than FSAE limit of {limit_pack_v}V**")
            self.red_flag += 1
        if self.cell_data["max_v"] / segment > limit_segment_v:
            print(f"**Segment Voltage greater than FSAE limit of {limit_segment_v}V**")
            self.red_flag += 1
        if round(self.cell_data["power"] / 1000) > limit_power:
            print(f"**Pack Power greater than FSAE limit of {limit_power}kW**")
            self.red_flag += 1
        if self.cell_data["peak_i_draw_percell"] > limit_i_draw_percell:
            print(f"Peak current draw per cell exceeds maximum rated current draw ({limit_i_draw_percell}A) for a single cell")
            self.red_flag += 1
        if self.cell_data["capacity"] > limit_capacity:
            print(f"Capacity of the pack is higher than necessary ({limit_capacity}Wh)")
            self.red_flag += 1
        if round(self.cell_data["length"] * 1.1, 1) > limit_length:
            print(f"Minimum pack length exceeds {limit_length}mm, and it is way too long.")
            self.red_flag += 1
        if self.red_flag == 0:
            print("**No warnings**")
        '''
        
        return self
    

        
    # get the voltage of a cell given the capacity and target_current for the drainage
    # if we know our initial voltage, next-step capacity, and current, this tells us the next voltage
    # then, we keep draining the capacity further at a different current, and we drop voltages again
    def get_cell_voltage(self, capacity, target_current):
        currents = [0.84, 4.2, 10, 20, 30] # constant-current values for our traces in Amps
        known_voltages = []

        for p in self.discharge_polynomials:
            # Predicted voltage at this capacity from each of the constant-current polynomials 
            known_voltages.append(p(capacity))
        
        voltage_interpolator = interp1d(currents, known_voltages, kind='cubic', fill_value='extrapolate')
        return voltage_interpolator(target_current)
    
    # allows us to set the discharge for new cars upon switching power caps
    def set_discharge(self, discharge_target):
        self.cell_data["discharge"] = discharge_target

    # update voltage and state of charge of the accumulator as we pull power from it
    def drain(self, power, time_step, last_current=-1):
        # assume we don't have a drain error yet
        #self.drain_error = False           # if we command a drain of ~0.0, this could be an issue


        # use P = IV to get the current through each cell
        total_current = power / self.cell_data["voltage"] #total current through the accumulator
        current_per_cell = total_current / self.parallel

        # update the energy of the accumulator (in kWh)
        energy_drained = power*time_step/3600                   # time is in seconds, power is in W  
        self.cell_data["discharge"] += energy_drained           # 1-capacity in Wh

        # assume all cells drain equally
        discharge_per_cell = self.cell_data["discharge"] / (self.parallel * self.pack_series)

        # divide by the *cell nominal voltage* to get capacity in mAh
        discharge_per_cell = 1000 * self.pack_series * discharge_per_cell / self.cell_data["nom_v"]

        # optional constraint on the rate of change of the current to minimzie effects of discontinuities in P(t)
        smoothing_bias = 0.3 # how much do we want the previous current to matter?
        if last_current > 0:
            target_current = smoothing_bias*last_current + (1-smoothing_bias)*current_per_cell
        else:
            target_current = current_per_cell

        # update the pack voltage according to the discharge curves
        #print("Discharge: ", discharge_per_cell)
        new_voltage = self.pack_series * self.get_cell_voltage(discharge_per_cell, target_current)
        if new_voltage > self.cell_data["min_v"]:
            self.cell_data["voltage"] = new_voltage
        else: 
            #print("Drain error: cells cannot drop below safe voltage")
            #self.cell_data["voltage"] = new_voltage
            # reset the drainage to cancel it out
            self.cell_data["discharge"] -= energy_drained           # 1-capacity in Wh
            self.drain_error = True

        # return the target current to smooth next time
        return target_current

    # TODO: how do we deal with a discontinuity in the P(t)? This causes a discontinuity in V(t) (usually an unphysical drop)

    # gets dV/dQ at our desired capacity and current
    def get_derivative(self, capacity, target_current):
        currents = [0.84, 4.2, 10, 20, 30] # constant-current values for our traces in Amps
        
        # calculate the numerical derivative of V with respect to Capacity for each curve
        derivatives = [p.deriv() for p in self.discharge_polynomials]
        known_derivatives = []

        for d in derivatives:
            # Predicted voltage at this capacity from each of the constant-current polynomials 
            known_derivatives.append(d(capacity))
        

        derivative_interpolator = interp1d(currents, known_derivatives, kind='cubic', fill_value='extrapolate')

        return derivative_interpolator(target_current)

    # predicts the next-step V(t) given some stimulus P(t) -- effectively numerically integrates V(Q-) for a given dQ-
    def new_drain(self, power, time_step, last_current=-1):
        # use P = IV to get the current through each cell
        total_current = power / self.cell_data["voltage"] #total current through the accumulator
        current_per_cell = total_current / self.parallel

        # update the energy of the accumulator (in kWh)
        energy_drained = power*time_step/3600                   # time is in seconds, power is in W  
        self.cell_data["discharge"] += energy_drained           # in Wh

        # get the capacity (Q) at each cell, assuming all cells drain equally
        discharge_per_cell = self.cell_data["discharge"] / (self.parallel * self.pack_series)

        # divide by the *cell nominal voltage* to get capacity in mAh; capacity here = total charge depleted from the cell
        discharge_per_cell = 1000 * self.pack_series * discharge_per_cell / self.cell_data["nom_v"]

        # get dQ-
        dQ_per_cell = energy_drained / (self.parallel * self.pack_series)                   # in Wh
        dQ_per_cell = 1000* self.pack_series * dQ_per_cell / self.cell_data["nom_v"]       # in mAh; convert using nominal cell voltage

        # get the differential change in voltage at our I(t) and Q(t) for one cell
        dV_dQ_cell = self.get_derivative(discharge_per_cell, current_per_cell)

        # update the pack voltage according to the discharge curves; multiply by pack_series to convert to pack dV
        # V = V0 + dV, where dV = dV/dQ- * dQ-
        new_voltage = self.cell_data["voltage"] + self.pack_series * dV_dQ_cell*dQ_per_cell

        if new_voltage > self.cell_data["min_v"]:
            self.cell_data["voltage"] = new_voltage
        else: 
            #print("Drain error: cells cannot drop below safe voltage")
            #self.cell_data["voltage"] = new_voltage
            # reset the drainage to cancel it out
            self.cell_data["discharge"] -= energy_drained           # 1-capacity in Wh
            self.drain_error = True


    def is_depleted(self):
        return self.drain_error
    
    def get_cell_data(self):
        return self.cell_data
    
    def set_drain_error(self, boolean):
        self.drain_error = boolean


def run_stats(series, parallel, segment):
    return Pack().pack(series, parallel, segment)




import matplotlib.pyplot as plt
import numpy as np
def test_pack(series, parallel, segment):
    currents = [0.84, 4.2, 10, 20, 30] # constant-current values for our traces in Amps
    pack = Pack()
    pack.pack(series, parallel, segment)        # .pack() initializes and resets the accumulator

    energy_to_drain = 6.9373299347852475 # kWh
    energy_to_drain_per_lap = energy_to_drain/22
    energy_to_drain_per_lap *= 3.6*10**6 # in Ws per lap

    dt = 55 #average laptime in s
    lap_power = energy_to_drain_per_lap/dt
    print("Lap Power: {:.2f}".format(lap_power))
    x = []
    y_star = []

    
    cell_data = pack.get_cell_data()
    print("Capacity: ", cell_data["capacity"])
    for i in range(32):
        pack.drain(lap_power, dt)

        cell_data = pack.get_cell_data()
        # check updated accumulator peak voltage
        y_star.append(cell_data["voltage"])
        x.append(cell_data["discharge"])

    print("Discharge: ", cell_data["discharge"])

    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    for i, p in enumerate(pack.discharge_polynomials):
        x = np.asarray(x)
        x1 = 1000 * x / (pack.cell_data["nom_v"] * pack.parallel)
        y = p(x1) #convert to discharge per cell 
        y = y*pack.pack_series
        plt.scatter(x, y, label='{}'.format(currents[i]), c= colors[i])


    plt.scatter(x, y_star, label='Trace')
    plt.plot(x, y_star)
    plt.xlabel('Discharge (Wh)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.legend()
    plt.show()

    

# 14, 4, 10
#test_pack(16, 5, 8)




"""
====================
Additional Functions
====================
"""
def check_valid_config():
    """
    Check to see what configuration is valid and didn't raise any red flag warnings.

    Note that the range in any of the three loops below can be set to anything custom.
    It is currently iterating through:
        12s -> 16s (16s is maximum we can go with custom BMS)
        4p -> 5p
        6 segment -> 10 segment
    """
    valid_config = []
    for ser in range(12, 17):
        for par in range(4,6):
            for seg in range(6,11):
                if run_stats(ser, par, seg).red_flag == 0:
                    valid_config.append([ser, par, seg])
    return sorted(valid_config)

"""
This line below prints all valid configurations
"""
#print(check_valid_config())

def highest_v_config():
    """
    Determines the configuration with the highest voltage
    **Doesn't find all configurations that has the same max voltage, so still need to fix that.
    """
    valid_config = check_valid_config()
    max_volt = 0
    config = []
    for i in valid_config:
        volt = run_stats(i[0], i[1], i[2]).cell_data["max_v"]
        if volt >= max_volt:
            max_volt = volt
            config = i
    return f"Max Pack Voltage from valid configuration is {max_volt} and is reached by configuration {config}"

"""
This line below finds the configuration with the highest voltage.
"""
#print(highest_v_config())
