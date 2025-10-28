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
import sys

import matplotlib.pyplot as plt
import numpy as np

import discharge_curves_energus
import discharge_curves_molicell
from scipy.interpolate import interp1d


class Molicel_Cell_21700:
    cell_data = {
        "name": "21700", 
        "weight": 70, # g
        "length": 70.2, # mm
        "min_v": 2.5, # V
        "nom_v": 3.6, # V
        "max_v": 4.2, # V
        "DCIR": 0.016, # Ohm
        "capacity": 15.5 # Wh
        }
    
    discharge_curve_currents = [0.84, 4.2, 10, 20, 30]


class Pack:
    def __init__(self, cell_type):
        self.cell_type = cell_type
    def pack(self, series, parallel, segment):
        cell_data = self.cell_type.cell_data

        pack_series = series * segment

        # Initialize pack data
        self.pack_series = pack_series
        self.parallel = parallel
        self.series = series

        self.pack_data = {}

        self.pack_data["name"] = f"{pack_series}s{parallel}p_Pack"
        self.pack_data["cell_count"] = parallel * pack_series

        self.pack_data["weight"] = round(cell_data["weight"] * (parallel * pack_series), 3) # g
        self.pack_data["length"] = round(cell_data["length"] * segment, 3) # mm
        self.pack_data["min_v"] = round(cell_data["min_v"] * pack_series, 3) # V
        self.pack_data["nom_v"] = round(cell_data["nom_v"] * pack_series, 3) # V
        self.pack_data["max_v"] = round(cell_data["max_v"] * pack_series, 3) # V

        self.pack_data["DCIR"] = round(parallel / (cell_data["DCIR"] * pack_series), 3)
        self.pack_data["capacity"] = round(cell_data["capacity"] * parallel * pack_series, 3) # Wh
        
        # Initialize voltage to max
        self.pack_data["voltage"] = self.pack_data["max_v"]

        # Initialize safe minimum voltage
        #self.pack_data["min_v"] = 400.0

        # Initialize total discharge from accumulator
        self.pack_data["discharge"] = 0
        self.discharge_polynomials = discharge_curves_molicell.return_polynomials()

        self.drain_error = False # have we over-drained this pack?
        self.breaker_popped = False # did we exceed our current max?

        self.derating_bound = 460
        self.safe_voltage = 400
        
        print("Creating a new pack of voltage {} V and capacity {} kWh".format(self.pack_data["voltage"], self.pack_data["capacity"] / 1000))
        return self

    # get the voltage of a cell given the capacity and target_current for the drainage
    # if we know our initial voltage, next-step capacity, and current, this tells us the next voltage
    # then, we keep draining the capacity further at a different current, and we drop voltages again
    def get_cell_voltage(self, capacity, target_current):
        currents = self.cell_type.discharge_curve_currents
        known_voltages = []

        for p in self.discharge_polynomials:
            # Predicted voltage at this capacity from each of the constant-current polynomials 
            known_voltages.append(p(capacity))
        
        voltage_interpolator = interp1d(currents, known_voltages, kind='cubic', fill_value='extrapolate')
        return voltage_interpolator(target_current)
    
    # allows us to set the discharge for new cars upon switching power caps
    def set_discharge(self, discharge_target):
        self.pack_data["discharge"] = discharge_target

    # update voltage and state of charge of the accumulator as we pull power from it
    def drain(self, power, time_step):
        # assume we don't have a drain error yet
        #self.drain_error = False           # if we command a drain of ~0.0, this could be an issue

        # use P = IV to get the current through each cell
        total_current = power / self.pack_data["voltage"] #total current through the accumulator
        current_per_cell = total_current / self.parallel

        # update the energy of the accumulator (in kWh)
        energy_drained = power*time_step/3600                   # time is in seconds, power is in W  
        self.pack_data["discharge"] += energy_drained           # 1-capacity in Wh

        # assume all cells drain equally
        discharge_per_cell = self.pack_data["discharge"] / (self.parallel * self.pack_series)

        # divide by the *cell nominal voltage* to get capacity in mAh
        discharge_per_cell = 1000 * self.pack_series * discharge_per_cell / self.pack_data["nom_v"]

        # update the pack voltage according to the discharge curves
        #print("Discharge: ", discharge_per_cell)
        new_voltage = self.pack_series * self.get_cell_voltage(discharge_per_cell, current_per_cell)
        if new_voltage > self.pack_data["min_v"]:
            self.pack_data["voltage"] = new_voltage
        else: 
            #print("Drain error: cells cannot drop below safe voltage")
            #self.pack_data["voltage"] = new_voltage
            # reset the drainage to cancel it out
            self.pack_data["discharge"] -= energy_drained           # 1-capacity in Wh
            self.drain_error = True

    # TODO: how do we deal with a discontinuity in the P(t)? This causes a discontinuity in V(t) (usually an unphysical drop)

    # gets dV/dQ at our desired capacity and current
    def get_derivative(self, capacity, target_current):
        currents = [0.84, 4.2, 10, 20, 30] # constant-current values for our traces in Amps; use this for Molicell
        # currents = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]   # Use this for Energus (SN3)

        # calculate the numerical derivative of V with respect to Capacity for each curve
        derivatives = [p.deriv() for p in self.discharge_polynomials]
        known_derivatives = []

        for d in derivatives:
            # Predicted voltage at this capacity from each of the constant-current polynomials 
            known_derivatives.append(d(capacity))

        derivative_interpolator = interp1d(currents, known_derivatives, kind='cubic', fill_value='extrapolate')

        return derivative_interpolator(target_current)

    #TODO throw current cap here also; add a "peak" and continuous current cap
    def set_derating(self, safe_minimum, derating_bound):
        self.safe_voltage = safe_minimum
        self.derating_bound = derating_bound

    # predicts the next-step V(t) given some stimulus P(t) -- effectively numerically integrates V(Q-) for a given dQ-
    # this will not drain anything if we pop our breaker; instead it will return the maximum current needed to stay safe
    def try_drain(self, power, time_step):
        """
        Drain the battery by a given power (W) over a time step (s).
        Updates cell voltage based on discharge curves.
        """

        # --- Compute pack voltage correctly ---
        pack_voltage = self.pack_data["voltage"]

        # Total current drawn from the pack
        total_current = power / pack_voltage  # in Amps
        current_per_cell = total_current / self.parallel

        # --- Current cap / breaker logic ---
        # current_cap = self.parallel * 45  # Max allowed current per parallel string
        # current_cap = 45 # A
        current_cap = 90 # A

        #print(current_per_cell)

        self.breaker_popped = False

        # Brownout protection: below 460 Volts, current cap -> TODO this was a hotfix for SN3
        # MAJOR TODO ----write a method to plot our curves and  
        # slope_for_derating = 5 * current_cap / 6

        # if self.pack_data["voltage"] < self.derating_bound:
        #     i1 = np.min([current_cap, np.max([10.0, current_cap*(slope_for_derating*(self.pack_data["voltage"]-self.derating_bound)+1)])])
        #     i2 = i1*self.safe_voltage/self.pack_data["voltage"]

        #     print("Derating the current to {} A at {} V".format(i2, self.pack_data["voltage"]))
            
        #     current_cap = i2

        # Throw a drain error if we pull more than the current cap
        if total_current > current_cap:
            self.breaker_popped = True
            target_current = current_cap
            return target_current, self.breaker_popped
        else:
            self.breaker_popped = False

            # --- Energy drained during this time step ---
            energy_drained_Wh = power * time_step / 3600  # W*s -> Wh
            self.pack_data["discharge"] += energy_drained_Wh  # Total pack energy drained

            # --- Energy per cell ---
            energy_per_cell_Wh = energy_drained_Wh / (self.pack_series * self.parallel)

            # Convert Wh to mAh for derivative calculation
            dQ_per_cell_mAh = 1000.0 * energy_per_cell_Wh / self.pack_data["nom_v"]


            # Current per cell in Amps already computed above
            dV_dQ_cell = self.get_derivative(
                1000.0 * (self.pack_data["discharge"] / (self.pack_series * self.parallel)) / self.pack_data["nom_v"],
                current_per_cell
            )
            print(1000.0 * (self.pack_data["discharge"] / (self.pack_series * self.parallel)) / self.pack_data["nom_v"])
            print(current_per_cell)
            print(dV_dQ_cell)
            print(  )
            print()

            # --- Update per-cell voltage ---
            new_cell_voltage = self.pack_data["voltage"] + dV_dQ_cell * dQ_per_cell_mAh

            # Enforce safe voltage limits
            if new_cell_voltage > self.pack_data["min_v"] and self.pack_data["discharge"] < self.pack_data["capacity"]:
                self.pack_data["voltage"] = new_cell_voltage
            else:
                # Rollback the energy drain if unsafe
                self.pack_data["discharge"] -= energy_drained_Wh
                self.drain_error = True

        return total_current, self.breaker_popped

    def is_depleted(self):
        return self.drain_error
    
    def get_pack_data(self):
        return self.pack_data
    
    def set_drain_error(self, boolean):
        self.drain_error = boolean

    # reload the pack to its initial state (i.e. reset discharge, voltage, and drain_error)
    def reset(self):
        self.set_discharge(0)
        self.pack_data["voltage"] = self.pack_data["max_v"]  
        self.drain_error = False
        self.breaker_popped = False

def run_stats(series, parallel, segment):
    return Pack().pack(series, parallel, segment)

def test_pack(series, parallel, segment):
    currents = [0.84, 4.2, 10, 20, 30] # constant-current values for our traces in Amps; use for Molicell

    # currents = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0] # use for energus


    pack = Pack()
    pack.pack(series, parallel, segment)        # .pack() initializes and resets the accumulator

    energy_to_drain = 4.5 # kWh
    energy_to_drain_per_lap = energy_to_drain/22
    energy_to_drain_per_lap *= 3.6*10**6 # in Ws per lap

    dt = 82 #average laptime in s
    lap_power = energy_to_drain_per_lap/dt
    print("Lap Power: {:.2f}".format(lap_power))
    x = []
    y_star = []


    
    pack_data = pack.get_pack_data()
    print("Capacity: ", pack_data["capacity"])
    for i in range(10):
        pack.try_drain(lap_power, dt)

        pack_data = pack.get_pack_data()
        # check updated accumulator peak voltage
        y_star.append(pack_data["voltage"])
        x.append(pack_data["discharge"])

    print("Discharge: ", pack_data["discharge"])

    #colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'magenta', 'pink', 'brown', 'black']

    for i, p in enumerate(pack.discharge_polynomials):
        x = np.asarray(x)
        x1 = 1000 * x / (pack.pack_data["nom_v"] * pack.parallel)
        y = p(x1) #convert to discharge per cell 
        y = y*pack.pack_series
        plt.scatter(x, y, label='{}A'.format(currents[i]), c= colors[i])


    plt.scatter(x, y_star, label='Our discharge')
    plt.plot(x, y_star)
    plt.title('Endurance Discharge for {}'.format(pack_data["name"]))
    plt.xlabel('Discharge (Wh)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.legend()
    plt.show()

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
    limit_pack_v = 600 #V
    limit_segment_v = 120 #V
    limit_power = 80 #kW
    limit_i_draw_percell = 45 #A
    limit_capacity = 10000 #Wh, capacity higher than 10000Wh is unnecessary.
    limit_length = 900 #mm, Pack will be too long beyond 900mm.

    valid_config = []
    for ser in range(12, 17):
        for par in range(4,6):
            for seg in range(6,11):
                test_pack = Pack(Molicel_Cell_21700)
                test_pack.pack(ser, par, seg)

                print(f"Testing pack ({ser}, {par}, {seg})")

                if test_pack.pack_data["max_v"] > limit_pack_v:
                    print(f"**Pack Peak Voltage greater than FSAE limit of {limit_pack_v}V**")
                    continue
                if test_pack.pack_data["max_v"] / test_pack.segment > limit_segment_v:
                    print(f"**Segment Voltage greater than FSAE limit of {limit_segment_v}V**")
                    continue
                if round(test_pack.pack_data["power"] / 1000) > limit_power:
                    print(f"**Pack Power greater than FSAE limit of {limit_power}kW**")
                    continue
                if test_pack.pack_data["peak_i_draw_percell"] > limit_i_draw_percell:
                    print(f"Peak current draw per cell exceeds maximum rated current draw ({limit_i_draw_percell}A) for a single cell")
                    continue
                if test_pack.pack_data["capacity"] > limit_capacity:
                    print(f"Capacity of the pack is higher than necessary ({limit_capacity}Wh)")
                    continue
                if round(test_pack.pack_data["length"] * 1.1, 1) > limit_length:
                    print(f"Minimum pack length exceeds {limit_length}mm, and it is way too long.")
                    continue
                print("Valid configuration")
                valid_config.append((ser, par, seg))
    return sorted(valid_config)

def highest_v_config():
    """
    Determines the configuration with the highest voltage
    **Doesn't find all configurations that has the same max voltage, so still need to fix that.
    """
    valid_config = check_valid_config()
    max_volt = 0
    config = []
    for i in valid_config:
        volt = run_stats(i[0], i[1], i[2]).pack_data["max_v"]
        if volt >= max_volt:
            max_volt = volt
            config = i
    return f"Max Pack Voltage from valid configuration is {max_volt} and is reached by configuration {config}"
