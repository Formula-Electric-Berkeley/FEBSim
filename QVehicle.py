import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

import powertrain

SPREADSHEET_PARAM_NAMES = [
    "Name", 
    "Total Mass", 
    "Tyre Radius"
]

# Return the first instance of a variable from a converted vehicle file.
def get_param(vehicle_df, param_name):
    return vehicle_df["Value"][vehicle_df["Description"] == param_name].iloc[0]

def get_params(vehicle_df, param_names):
    return [get_param(vehicle_df, param_name) for param_name in param_names]

def vehicle_from_file(vehicle_file):
    veh = Vehicle()
    

class Vehicle:
    @staticmethod
    def from_file(vehicle_file):
        details_df = pd.io.excel.read_excel(vehicle_file, sheet_name="Info")

        torque_curve_df = pd.io.excel.read_excel(vehicle_file, sheet_name="Torque Curve")
        torque_function = scipy.interpolate.interp1d(torque_curve_df["Engine Speed [rpm]"], torque_curve_df["Torque [Nm]"])

        return Vehicle.from_dict(
            dict(zip(SPREADSHEET_PARAM_NAMES, get_params(details_df, SPREADSHEET_PARAM_NAMES))), 
            torque_function
        )

    @staticmethod
    def from_dict(vehicle_dict, torque_curve):
        veh = Vehicle()
        veh.name, veh.mass, veh.tire_radius = vehicle_dict["Name"], vehicle_dict["Total Mass"], vehicle_dict["Tyre Radius"]  # God I hate the british spelling so bad

        print("New vehicle instantiated:", veh.get_params_dict())

        veh.torque_function = torque_curve  

        return veh

    def get_params_dict(self):
        return dict(zip(
            SPREADSHEET_PARAM_NAMES,
            [self.name, self.mass, self.tire_radius]
        ))

    def copy_with_params(self, changed_params):
        params = self.get_params_dict()

        for param_name, param_value in changed_params.items():
            if param_name not in params:
                raise Exception("This param is not stored in this vehicle. Remove this if we want to later I guess?")

            params[param_name] = param_value

        return Vehicle.from_dict(params, torque_curve)

