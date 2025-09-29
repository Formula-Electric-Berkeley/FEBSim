'''
More advanced model for laptime simulation. 
'''
import os
import time
import numpy as np
import casadi as ca
import pandas as pd
import vehicle as veh
import powertrain_model
import track_and_car as tc
from scipy.interpolate import interp1d
from track_for_bicycle import Track
from colorama import Fore, Style



def save_output(data, filename, directory='sims_logs/', metadata=None):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Define the file path
    file_path = os.path.join(directory, filename)
    
    # Check if the file exists, and if so, increment the number in the filename
    base_name, extension = os.path.splitext(filename)
    counter = 1
    
    while os.path.exists(file_path):
        # Increment the counter and update the filename
        new_filename = f"{base_name}_{counter}{extension}"
        file_path = os.path.join(directory, new_filename)
        counter += 1
        
    print(file_path)    


    # Write the dataframe to a excel sheet
    # with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
    #     data.to_excel(writer, sheet_name="data", index=False)
    #     if metadata is not None:
    #         metadata.to_excel(writer, sheet_name="metadata", index=False)


class Vehicle:
    def __init__(self, **kwargs):
        # Initialize the vehicle with 30kW SN3 and allow for altered parameters
        self.params = {"M": veh.M, "lf": veh.lf, "lr": veh.lr, "wf": veh.wf, "wr": veh.wr, 
                        "Je": veh.Je, "Jw" : veh.Jw, "re": veh.re, "rb": veh.rb, "Iz": veh.Iz, "air_density": veh.rho, 
                        "A": veh.A, "max_velocity": veh.max_velocity, "brake_fr": veh.brake_fr, "brake_fl": veh.brake_fl, 
                        "brake_rr": veh.brake_rr, "brake_rl": veh.brake_rl, "gear_ratio": veh.gear_ratio, "ratio_final": veh.ratio_final, 
                        "delta_max": veh.delta_max, "drive_max": veh.drive_max, "brake_max": veh.brake_max, "cg_height": veh.cg_height, "rho": veh.rho, 
                        "Cl": veh.Cl, "Cd": veh.Cd, 
                        # Tire model
                        "mu": 0.75,  "grip_factor": 1, "Bx": 16, "Cx": 1.58, "Ex": 0.1, "By": 13, "Cy": 1.45, "Ey": -0.8}
        
    
        # Allow parameter overrides
        self.params.update(kwargs)
        self.initialize_tire_model()


    # Dynamically update vehicle params
    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                print("\033[93m" + f"\nWarning: {key} is not a recognized parameter." + "\033[0m")

    
    def sweep_parameters(self, param_ranges):
        """
        Sweep over multiple parameter combinations.
        
        param_ranges: dict
            Dictionary where keys are parameter names and values are lists of values to test.
        
        Returns:
            List of vehicle instances with different parameter sets.
        """
        keys = list(param_ranges.keys())
        values_list = list(param_ranges.values())

        # Generate all combinations of parameter values manually
        def generate_combinations(index=0, current_params={}):
            if index == len(keys):
                new_vehicle = Vehicle(**current_params)
                vehicles.append(new_vehicle)
                return
            
            key = keys[index]
            for value in values_list[index]:
                current_params[key] = value
                generate_combinations(index + 1, current_params.copy())

        vehicles = []
        generate_combinations()
        return vehicles

    def initialize_tire_model(self):
        SL = ca.MX.sym("SL")
        SA = ca.MX.sym("SA")
        Fz = ca.MX.sym("Fz")

        self.Fz_bins = [0.5*(436.75-41.51), 0.5*(436.75+673.72), 0.5*(673.72+1002.38), 0.5*(1002.38+1264.22)]
        self.B_vals = [-0.28, -0.22, -0.16, -0.13]
        self.C_vals = [1.05, 1.5, 1.7, 1.73]
        self.D_vals = [800, 1300, 2000, 2500]
        self.E_vals = [1.15, 0.4, 0.36, 0.38]
        self.F_vals = [0.023, 0.038, 0.035, 0.01]

        self.B_interp = ca.interpolant('B', 'bspline', [self.Fz_bins], self.B_vals)
        self.C_interp = ca.interpolant('C', 'bspline', [self.Fz_bins], self.C_vals)
        self.D_interp = ca.interpolant('D', 'bspline', [self.Fz_bins], self.D_vals)
        self.E_interp = ca.interpolant('E', 'bspline', [self.Fz_bins], self.E_vals)
        self.F_interp = ca.interpolant('F', 'bspline', [self.Fz_bins], self.F_vals)

        By = self.B_interp(Fz)
        Cy = self.C_interp(Fz)
        Dy = self.D_interp(Fz)
        Ey = self.E_interp(Fz)
        Fy = self.F_interp(Fz)

        ForceX = 0
        ForceY = Dy * ca.sin(Cy * ca.atan(By * SA - Ey * (By * SA - ca.atan(By * SA))) + Fy)

        self.tire_model_func = ca.Function("tire_model_func", [SL, SA, Fz], [ForceX, ForceY])


    def combined_slip_forces(self, SL, SA, Fn):
        # Convert radians to angles
        SA = (SA * 180) / np.pi
        SL = (SL * 180) / np.pi
        
        # Get forces pacejka model
        
        By, Cy, Dy, Ey, Fy = [
            -0.34876850845497015 - 0.00034276883261057686 * Fn - 0.00000013247032127260662 * (Fn ** 2),
            0.5509222176106313 - 0.002443688073927094 * Fn - .0000012320536839299613 * (Fn ** 2),
            338.39870577325456 - 1.9769442425898722 * Fn,
            0.3553895949382016 + 31.244897244705058 * ca.exp(0.016405921135532898 * Fn),
            -0.0233809460004577 - 0.00017739222796836902 * Fn - 0.00000012122598706268264 * (Fn ** 2)
        ]

        ForceY = Dy * ca.sin(Cy * ca.atan(By * SA - Ey * (By * SA - ca.atan(By * SA))) + Fy)

        Bx, Cx, Dx, Ex, Fx = [
            -17.3662 - 9.50547 * ca.exp(0.00226287 * Fn),
            -0.00127868 + 0.0000135334 * Fn + 0.00000000486905 * (Fn ** 2),
            43690.6 - 173.535 * Fn - 0.0708504 * (Fn ** 2),
            -1.42217 + 0.0015213 * Fn,
            0.000176214 - 0.000000274311 * Fn
        ]

        ForceX = Dx * ca.sin(Cx * ca.atan(Bx * SL - Ex * (Bx * SL - ca.atan(Bx * SL))) + Fx)

        ForceX = ForceX*self.params["grip_factor"]
        ForceY = ForceY*self.params["grip_factor"]

        return ForceX, ForceY



class BicycleModel:
    def __init__(self, track_file, mesh_size, track_parts):
        # Initialize the track and vehicle by loading the relevant files
        self.initialize_track(track_file, mesh_size, track_parts)
        self.initialize_vehicle()
        self.track_file = track_file
        self.mesh_size = mesh_size

    def initialize_track(self, trackfile, mesh_size, parts):
        self.tr = Track(trackfile, mesh_size)
        print("\033[1;92m" + "\nINITIALIZED TRACK: {}".format(trackfile) + "\033[0m")
        
        # Break the track into parts splines of curvature
        # print(self.tr.get_length())
        
        self.x_parts, self.parts, self.indices_to_partition = self.tr.split_on_straights(self.tr.get_length() // 10, 3, 8)
        # self.tr.plot()
        # self.tr.plot_track_segments(self.x_parts, self.parts, self.indices_to_partition)

        self.n_max_unnormalized = 4
        if 'skidpad' in trackfile:
            self.n_max_unnormalized = 1.2

    def initialize_vehicle(self):
        self.motor = powertrain_model.motor()
        self.diff = powertrain_model.drexler_differential()
        self.vehicle = Vehicle()
        self.veh = self.vehicle.params # save off our vehicle parameters for convenience

    def update_vehicle(self, veh):
        self.vehicle = veh
        self.veh = veh.params

    def run(self, simple=False):        
        dfs = []

        init_state = None
        
        for i, part in enumerate(self.parts[:1]):
            print("\033[1;92m" + "\nRUNNING PART {}: ".format(i) + "\033[0m")
            
            # Pass init_state so it continues from the last state's final values
            if simple:
                df_segment = self.opt_mintime_simple(part, self.mesh_size, init_state=init_state, segment_number = i)
            else:
                df_segment = self.opt_mintime(part, self.mesh_size, init_state=init_state, segment_number = i)
            dfs.append(df_segment)
            
            # If there's a solution, store the final state for the next segment
            if not df_segment.empty:
                last_row = df_segment.iloc[-1]
                print("\033[1;94m" + "\nLAST ROW:" + "\033[0m")
                print(last_row)
                
                if simple:
                    init_state = {
                        "v": last_row["v"],
                        "beta": last_row["beta"],
                        "omega_z": last_row["omega_z"],
                        "n": last_row["n"],
                        "xi": last_row["xi"],
                        "delta": last_row["delta"],
                        "torque_drive": last_row["torque_drive"],
                        "f_brake": last_row["f_brake"],
                        "gamma_y": last_row["gamma_y"]
                    }  

                else:
                    init_state = {
                        "v": last_row["v"],
                        "beta": last_row["beta"],
                        "omega_z": last_row["omega_z"],
                        "n": last_row["n"],
                        "xi": last_row["xi"],
                        "wfr": last_row["wfr"],
                        "wfl": last_row["wfl"],
                        "wrl": last_row["wbl"],
                        "wrr": last_row["wbr"]
                    }


        combined_df = pd.concat(dfs, axis=0, ignore_index=True)

        output_file_name = self.track_file[:-5] + '_' + f'{time.time()}' + '.xlsx'
        print("got to here")
        save_output(combined_df, output_file_name)

        tc.plot_track(self.track_file, output_file_name, self.mesh_size)

        laptime = sum(df["time"].iloc[-1] for df in dfs if not df.empty)

        energy = sum(df["energy"].iloc[-1] for df in dfs if not df.empty)

        print("\033[1;92" + '\nTotal laptime is {:.2f} seconds with an energy consumption of {:.2f} kWh'.format(laptime, energy) + "\033[0m")

        return laptime, energy

   
    def opt_mintime_simple(self, curvatures, mesh_size, init_state=None, segment_number = None):
        # ------------------------------------------------------------------------------------------------------------------
        # GET TRACK INFORMATION --------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # get the curvature of the track centerline at each mesh point
        kappa_refline_cl = curvatures

        # the total number of mesh points
        num_mesh_points = len(curvatures)

        # optimization steps (0, 1, 2 ... end point)
        steps = [i for i in range(num_mesh_points)]
        N = steps[-1]

        # separation between each mesh point (in the future, this could be vary between points!)
        mesh_point_separations = np.ones(num_mesh_points) * mesh_size

        # interpolate track data from reference line to the number of steps sampled
        kappa_interp = ca.interpolant("kappa_interp", "linear", [steps], kappa_refline_cl)

        # ------------------------------------------------------------------------------------------------------------------
        # DIRECT GAUSS-LEGENDRE COLLOCATION --------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        """
        In this section, we represent our state and command trajectories as a spline
        We construct our spline using Lagrange polynomials to make interpolation easier
        """
        # degree of interpolating polynomial ("sweet spot" according to MIT robotics paper)
        d = 3

        # legendre collocation points ; gives the temporal location of the collocation points along our spline
        tau = np.append(0, ca.collocation_points(d, "legendre"))

        # coefficient matrix for the collocation equation
        C = np.zeros((d + 1, d + 1))

        # coefficient matrix for the continuity equation
        D = np.zeros(d + 1)

        # coefficient matrix for the quadrature function
        B = np.zeros(d + 1)

        # construct polynomial basis using Lagrange polynomials
        for j in range(d + 1):
            # construct Lagrange polynomials to get the polynomial basis at each collocation point
            p = np.poly1d([1])
            for r in range(d + 1):
                if r != j:
                    # Lagrange polynomial coefficient for interpolating between collocation points
                    p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

            # evaluate polynomial at the final time to get the coefficients of the continuity equation
            # this helps ensure continuity between consecutive finite elements
            D[j] = p(1.0)

            # evaluate time derivative of polynomial at collocation points to get the coefficients of collocation equation
            p_der = np.polyder(p)
            for r in range(d + 1):
                C[j, r] = p_der(tau[r])

            # evaluate integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            # this is used for approximating the integral of Lagrange polynomials over a finite element
            B[j] = pint(1.0)

        # ------------------------------------------------------------------------------------------------------------------
        # STATE VARIABLES --------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        """
        In the next two sections, we define our state and control variables, as well as their (approximate) relative magnitudes 
        CasADi's NLP converges best when we give it normalized input variables because our problem spans several orders of magnitude
        (e.g. max torque is 5500 Nm, max steer is 0.66 rad)
        """

        # number of state variables
        nx = 5

        # velocity [m/s]
        v_n = ca.MX.sym(
            "v_n"
        )  # we give CasADi's solver normalized variables to improve convergence
        v_s = 30  # scale the normalized vector
        v = v_s * v_n

        # sideslip angle [rad/s]
        beta_n = ca.MX.sym("beta_n")
        beta_s = 1.0
        beta = beta_s * beta_n

        # yaw rate [rad/s]
        omega_z_n = ca.MX.sym("omega_z_n")
        omega_z_s = 1
        omega_z = omega_z_s * omega_z_n

        # lateral distance to reference line (positive = left) [m]
        n_n = ca.MX.sym("n_n")
        n_s = 4.0
        n = n_s * n_n

        # relative angle to tangent on reference line [rad]
        xi_n = ca.MX.sym("xi_n")
        xi_s = 1.0
        xi = xi_s * xi_n

        x_s = np.array([v_s, beta_s, omega_z_s, n_s, xi_s])

        # vertical vector for the state variables
        x = ca.vertcat(v_n, beta_n, omega_z_n, n_n, xi_n)

        # ------------------------------------------------------------------------------------------------------------------
        # CONTROL VARIABLES ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # number of control variables
        nu = 4

        # steer angle [rad]
        delta_n = ca.MX.sym("delta_n")
        delta_s = 0.5
        delta = delta_s * delta_n

        # positive drive torque from the motor [N-m]
        torque_drive_n = ca.MX.sym("f_drive_n")
        torque_drive_s = 600.0
        torque_drive = torque_drive_s * torque_drive_n

        # negative longitudinal force (brake) [N]
        f_brake_n = ca.MX.sym("f_brake_n")
        f_brake_s = 20000.0
        f_brake = f_brake_s * f_brake_n

        # total lateral wheel load transfer [N]
        gamma_y_n = ca.MX.sym("gamma_y_n")
        gamma_y_s = 5000.0
        gamma_y = gamma_y_s * gamma_y_n

        # curvature of reference line [rad/m] (allows CasADi to interpret kappa in our math below)
        kappa = ca.MX.sym("kappa")

        # scaling factors for control variables
        u_s = np.array([delta_s, torque_drive_s, f_brake_s, gamma_y_s])

        # put all controls together
        u = ca.vertcat(delta_n, torque_drive_n, f_brake_n, gamma_y_n)

        # ------------------------------------------------------------------------------------------------------------------
        # MODEL PHYSICS ----------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        """
        In this section, we define the physical model equations that constrain our system
        The physical equations here ultimately define how our state variables evolve over time (and thus over the lap)

        TODO: ensure the physics is correct -> use Ackermann geometry
        """


        # define physical constants and vehicle parameters; pass these in in the future
        g = 9.81  # [m/s^2]
        mass = self.veh["M"]    # [kg]
        lf = self.veh["lf"]     # front wheelbase length (from COM to front axle)
        lr = self.veh["lr"]     # rear  wheelbase length
        L = lf + lr     # total wheelbase length
        wf = self.veh["wf"]     # front axle width (distance between centers of wheels)
        wr = self.veh["wr"]     # rear axle width

        Je = self.veh["Je"]     # engine moment of inertia TODO -> modify with real data
        Jw = self.veh["Jw"]     # wheel moment of inertia
        re = self.veh["re"]     # effective wheel radius
        rb = self.veh["rb"]     # effective brake radius
        Iz = self.veh["Iz"]     # vehicle moment of inertia about z axis

        # Aero stuff****************************
        air_density = self.veh["rho"]
        frontal_area = self.veh["A"]
        Cl = self.veh["Cl"]      # NEGATIVE Cl means downforce
        Cd = self.veh["Cd"]      # negative Cd means drag

        drag_coeff = 0.5*air_density*Cd*frontal_area
        df_coeff = - 0.5*air_density*Cl*frontal_area    # "Positive force" = down

        # Compute aerodynamic and drag forces
        # Assumes df shared equally bw all wheels and that front area and Cl/Cd do not change w angle of attack
        # Also assumes 0.5 COP right now; this can change later
        Fx_aero_drag = drag_coeff * v * v
        Fz_aero_down = df_coeff * v * v          
        
        # compute normal forces at each wheel, assumes wf roughly equals wr
        # this will be used by our tire model to compute grip forces at each wheel
        # 50/50 left-right wb; forward-back wb dependent on lf, lr

        k_roll = 0.5        # roll balance: how is gamma_y distributed between front and back
        k_pitch = 0.5       # how is longitudinal load difference distributed between left and right 

        # static normal tire forces [N]
        F_Nfr_weight = 0.5*(mass * g) * lr / L 
        F_Nfl_weight = 0.5*(mass * g) * lr / L
        F_Nrr_weight = 0.5*(mass * g) * lf / L 
        F_Nrl_weight = 0.5*(mass * g) * lf / L 

        # dynamic normal tire forces (load transfer) [N]
        F_Nfr_dyn = -k_roll*gamma_y
        F_Nfl_dyn =  k_roll*gamma_y
        F_Nrr_dyn = -1*(1-k_roll)*gamma_y
        F_Nrl_dyn = (1-k_roll)*gamma_y

        # sum of all normal tire forces (including aero) [N]
        F_Nfr = F_Nfr_weight + F_Nfr_dyn + Fz_aero_down*0.25
        F_Nfl = F_Nfl_weight + F_Nfl_dyn + Fz_aero_down*0.25
        F_Nrr = F_Nrr_weight + F_Nrr_dyn + Fz_aero_down*0.25
        F_Nrl = F_Nrl_weight + F_Nrl_dyn + Fz_aero_down*0.25

        # project velocity to longitudinal and lateral vehicle axes (for eventual calculation of drag and slip angle/ratio)
        vx = v * ca.cos(beta)
        vy = v * ca.sin(beta)

        # compute body-frame components of linear velocity at each wheel
        vx_fr = vx + wf / 2 * omega_z
        vy_fr = vy + lf * omega_z
        vx_fl = vx - wf / 2 * omega_z
        vy_fl = vy + lf * omega_z
        vx_rr = vx + wr / 2 * omega_z
        vy_rr = vy - lr * omega_z
        vx_rl = vx - wr / 2 * omega_z
        vy_rl = vy - lr * omega_z

        # compute wheel-frame velocity components (slip ratio/angle are defined in the frame of each wheel)
        vlfr = vx_fr * ca.cos(delta) + vy_fr * ca.sin(delta)
        vtfr = vy_fr * ca.cos(delta) - vx_fr * ca.sin(delta)
        vlfl = vx_fl * ca.cos(delta) + vy_fl * ca.sin(delta)
        vtfl = vy_fl * ca.cos(delta) - vx_fl * ca.sin(delta)
        vlrr = vx_rr
        vtrr = vy_rr
        vlrl = vx_rl
        vtrl = vy_rl

        # compute the slip angles from the velocities at each wheel
        epsilon = 1e-3
        # alpha_fr = -ca.arctan(vtfr / (ca.fabs(vlfr)  + epsilon))
        # alpha_fl = -ca.arctan(vtfl / (ca.fabs(vlfl) + epsilon))
        # alpha_rr = -ca.arctan(vtrr / (ca.fabs(vlrr) + epsilon))
        # alpha_rl = -ca.arctan(vtrl / (ca.fabs(vlrl) + epsilon))

        # slip angles [rad]
        alpha_fl = delta - ca.atan((vy_fl) /
                                (vx_fl))
        alpha_fr = delta - ca.atan((vy_fr) /
                                (vx_fr))
        
        # identical for rear wheels
        alpha_rl = -ca.atan((vy_rl) /
                        (vx_rl))
        alpha_rr = -ca.atan((vy_rr) /
                        (vx_rr))

        # compute wheel grip forces in the wheel frame
        # Flfr, Ftfr = self.vehicle.tire_model_func(ca.MX(0.0), alpha_fr, F_Nfr)  
        # Flfl, Ftfl = self.vehicle.tire_model_func(ca.MX(0.0), alpha_fl, F_Nfl)
        # Flrr, Ftrr = self.vehicle.tire_model_func(ca.MX(0.0), alpha_rr, F_Nrr)
        # Flrl, Ftrl = self.vehicle.tire_model_func(ca.MX(0.0), alpha_rl, F_Nrl)
        __, f_trans_fr = self.vehicle.combined_slip_forces(0.0, alpha_fr, F_Nfr)  
        __, f_trans_fl = self.vehicle.combined_slip_forces(0.0, alpha_fl, F_Nfl)
        __, f_trans_rr = self.vehicle.combined_slip_forces(0.0, alpha_rr, F_Nrr)
        __, f_trans_rl = self.vehicle.combined_slip_forces(0.0, alpha_rl, F_Nrl)

        # distribute commanded brake force to the wheels based on our vehicle's brake force distribution
        F_brake_fr = self.veh["brake_fr"] * f_brake  # brake force of front right wheel
        F_brake_fl = self.veh["brake_fl"] * f_brake
        F_brake_rl = self.veh["brake_rl"] * f_brake
        F_brake_rr = self.veh["brake_rr"] * f_brake

        # longitudinal tire forces based on simplest longitudinal model [N]
        # Torque_drive is the torque output by the motor, so wheels receive a greater torque
        f_drive = self.veh["gear_ratio"] * torque_drive / (re)
        f_long_fr = F_brake_fr
        f_long_fl = F_brake_fl
        f_long_rl = 0.5 * f_drive + F_brake_rl
        f_long_rr = 0.5 * f_drive + F_brake_rr

        # change wheel forces to body frame
        f_x_fr = f_long_fr * ca.cos(delta) - f_trans_fr * ca.sin(delta)
        f_y_fr = f_long_fr * ca.sin(delta) + f_trans_fr * ca.cos(delta)
        f_x_fl = f_long_fl * ca.cos(delta) - f_trans_fl * ca.sin(delta)
        f_y_fl = f_long_fl * ca.sin(delta) + f_trans_fl * ca.cos(delta)
        f_x_rr = f_long_rr
        f_y_rr = f_trans_rr
        f_x_rl = f_long_rl    # Driven wheels are oriented parallel to the car in RWD
        f_y_rl = f_trans_rl

        
        # compute the net force and torque on the vehicle
        Fx = (
            f_x_fr + f_x_fl + f_x_rl + f_x_rr + Fx_aero_drag
        )  # net force in the x (longitudinal car direction)
        Fy = f_y_fr + f_y_fl + f_y_rr + f_y_rl  # net force in the y (lateral car direction)


        # Compute net torque about Z axis
        Kz = (
            lf * (f_y_fr + f_y_fl)
            - lr * (f_y_rr + f_y_rl)
            + wf / 2 * (f_x_fr - f_x_fl)
            + wr / 2 * (f_x_rr - f_x_rl)
        )  

        # in their convention, fx = f_long, fy = f_trans; these are equivalent statements
        # Kz = ((f_x_rr - f_x_rl) * wr / 2
        #         - (f_y_rl + f_y_rr) * lr
        #         + ((f_x_fr - f_x_fl) * ca.cos(delta)
        #             + (f_y_fl - f_y_fr) * ca.sin(delta)) * wf / 2
        # This should be lf, but otherwise they're identical still.
        # The moment is about Iz; y component of fr and fr wheels moves lf back to COM           
        #         + ((f_y_fl + f_y_fr) * ca.cos(delta)
        #             + (f_x_fl + f_x_fr) * ca.sin(delta)) * wf)

        # ------------------------------------------------------------------------------------------------------------------
        # DERIVATIVES ------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        """
        In this section, we define the rates of change of each of our state variables
        Based on the UMunich paper, we reformulate the problem to integrate longitudinal distance traveled relative to the refline
        """

        # time-distance scaling factor (dt/ds) -> this is just the chain rule!
        sf = (1.0 - n * kappa) / (v * (ca.cos(xi + beta)))
        # assume this is non-negative for now; we must define n and kappa more rigorously later
        # TODO: hard prove everything from UMunich by hand; do later

        # vx_dot =  (vy*omega_z + Fx/mass)*sf #includes Newton-Euler pseudoforces due to rotation
        # vy_dot = (-vx*omega_z + Fy/mass)*sf

        dv = (sf / mass) * (
            (f_long_rl + f_long_rr) * ca.cos(beta)
            + (f_long_fl + f_long_fr) * ca.cos(delta - beta)
            + (f_trans_rl + f_trans_rr) * ca.sin(beta)
            - (f_trans_fl + f_trans_fr) * ca.sin(delta - beta)
            - Fx_aero_drag * ca.cos(beta)
        )

        dbeta = sf * (
            -omega_z
            + (
                -(f_long_rl + f_long_rr) * ca.sin(beta)
                + (f_long_fl + f_long_fr) * ca.sin(delta - beta)
                + (f_trans_rl + f_trans_rr) * ca.cos(beta)
                + (f_trans_fl + f_trans_fr) * ca.cos(delta - beta)
                + Fx_aero_drag * ca.sin(beta)
            )
            / (mass * v)
        )

        # angular acceleration about the z axis
        omegaz_dot = sf * Kz / Iz

        xi_dot = sf * omega_z - kappa
        n_dot = sf * v * ca.sin(xi + beta)

    
        # define our driving dynamics with a series of ordinary differential equations
        dx = (
            ca.vertcat(
                dv, dbeta, omegaz_dot, n_dot, xi_dot
            )
            / x_s
        )

        print("\033[1;92m" + "\nDYNAMICS DEFINED" + "\033[0m")        
        # ------------------------------------------------------------------------------------------------------------------
        # CONTROL BOUNDARIES -----------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        """
        In the next two sections, we set the min and max values for each of the control and state variables
        The model will ensure these inequalities are satisfied at every collocation point
        Making sure these boundaries are physical (not too broad, not too narrow) is *crucial* for accurate convergence 
        Because we give CasADi normalized vectors, we must normalize these bounds by dividing by each variable's scale x_s
        """

        delta_min = -self.veh["delta_max"] / delta_s  # min. steer angle [rad]
        delta_max = self.veh["delta_max"] / delta_s  # max. steer angle [rad]

        f_drive_min = 0.0  # min. longitudinal drive TORQUE [Nm]
        f_drive_max = self.veh["drive_max"] / torque_drive_s  # max. longitudinal drive torque [Nm]

        # the value of our brake force is always negative
        f_brake_min = -self.veh["brake_max"] / f_brake_s  # min. longitudinal brake force [N]
        f_brake_max = 0.0  # max. longitudinal brake force [N]

        # The load transfer forces are unbounded; later we constrain them by enforcing no roll / pitch
        gamma_y_min = -np.inf  # min. lateral wheel load transfer [N]
        gamma_y_max = np.inf  # max. lateral wheel load transfer [N]

        # ------------------------------------------------------------------------------------------------------------------
        # STATE BOUNDARIES -------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        v_min = 3.0 / v_s  # min. velocity [m/s]
        v_max = self.veh["max_velocity"] / v_s  # max. velocity [m/s]

        # Note: it is crucial beta + xi cannot equal pi/2; 
        # beta = 90 does NOT mean pure lateral acceleration; it means the car drives sideways
        beta_min = -0.25 * np.pi / beta_s  # min. side slip angle [rad]
        beta_max = 0.25 * np.pi / beta_s  # max. side slip angle [rad]

        omega_z_min = -0.5 * np.pi / omega_z_s  # min. yaw rate [rad/s]
        omega_z_max = 0.5 * np.pi / omega_z_s  # max. yaw rate [rad/s]

        # we need a moderate xi when starting to corner, but too large causes convergence issues
        xi_min = (
            -0.25 * np.pi / xi_s
        )  # min. relative angle to tangent of reference line [rad]
        xi_max = (
            0.25 * np.pi / xi_s
        )  # max. relative angle to tangent of reference line [rad]

        # an average f1 track is 12m wide -1.5m for the car, -1m buffer on each side
        n_min = -self.n_max_unnormalized / n_s  # min lateral distance from reference line [m]
        n_max = self.n_max_unnormalized / n_s

        # ------------------------------------------------------------------------------------------------------------------
        # INITIAL GUESS FOR DECISION VARIABLES -----------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        vx_guess = 15.0 / v_s

        # ------------------------------------------------------------------------------------------------------------------
        # HELPER FUNCTIONS -------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # continuous time dynamics
        f_dyn = ca.Function(
            "f_dyn", [x, u, kappa], [dx, sf], ["x", "u", "kappa"], ["dx", "sf"]
        )

        # longitudinal tire forces in the vehicle frame [N]
        f_fx = ca.Function(
            "f_fx",
            [x, u],
            [f_x_fl, f_x_fr, f_x_rl, f_x_rr],
            ["x", "u"],
            ["f_x_fl", "f_x_fr", "f_x_rl", "f_x_rr"],
        )
        # lateral tire forces in the vehicle frame [N]
        f_fy = ca.Function(
            "f_fy",
            [x, u],
            [f_y_fl, f_y_fr, f_y_rl, f_y_rr],
            ["x", "u"],
            ["f_y_fl", "f_y_fr", "f_y_rl", "f_y_rr"],
        )

        # vertical tire forces [N]
        f_fz = ca.Function(
            'f_fz', 
            [x, u], 
            [F_Nfl, F_Nfr, F_Nrl, F_Nrr],
            ['x', 'u'], 
            ['f_z_fl', 'f_z_fr', 'f_z_rl', 'f_z_rr']
        )


        # ------------------------------------------------------------------------------------------------------------------
        # FORMULATE NONLINEAR PROGRAM --------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # initialize NLP vectors
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # initialize ouput vectors
        x_opt = []
        u_opt = []
        dt_opt = []
        k_opt = []
        ec_opt = []

        # initialize control vectors (for regularization)
        delta_p = []
        F_p = []

        # boundary constraint: set initial conditions for the state variables
        Xk = ca.MX.sym("X0", nx)
        w.append(Xk)
        lbw.append(
            [
                v_min,
                beta_min,
                omega_z_min,
                n_min,
                xi_min
            ]
        )
        ubw.append(
            [
                v_max,
                beta_max,
                omega_z_max,
                n_max,
                xi_max
            ]
        )
        w0.append(
            [
                vx_guess,
                0.0,
                0.0,
                0.0,
                0.0
            ]
        )
        x_opt.append(Xk * x_s)

        # If init_state is given, fix the initial state to that
        if init_state is not None:
            init_v = init_state["v"] / v_s
            init_beta = init_state["beta"] / beta_s
            init_omega_z = init_state["omega_z"] / omega_z_s
            init_n = init_state["n"] / n_s
            init_xi = init_state["xi"] / xi_s
            init_vec = [
                init_v,
                init_beta,
                init_omega_z,
                init_n,
                init_xi
            ]
            lbw[-1] = init_vec
            ubw[-1] = init_vec
            w0[-1] = init_vec

        # loop along the racetrack and formulate path constraints & system dynamics
        for k in range(N):
            # add decision variables for the control
            Uk = ca.MX.sym("U_" + str(k), nu)
            w.append(Uk)
            lbw.append([delta_min, f_drive_min, f_brake_min, gamma_y_min])
            ubw.append([delta_max, f_drive_max, f_brake_max, gamma_y_max])
            w0.append([0.0] * nu)

            # add decision variables for the state at collocation points
            Xc = []
            for j in range(d):
                Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), nx)
                Xc.append(Xkj)
                w.append(Xkj)

                lbw.append([
                    v_min,
                    beta_min,
                    omega_z_min,
                    n_min,
                    xi_min
                ])
                ubw.append([
                    v_max,
                    beta_max,
                    omega_z_max,
                    n_max,
                    xi_max
                ])
                w0.append(
                    [
                        vx_guess,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ]
                )

            # loop over all collocation points
            Xk_end = D[0] * Xk
            sf_opt = []
            curv = []
            for j in range(1, d + 1):
                # calculate the state derivative at the collocation point
                xp = C[0, j] * Xk
                for r in range(d):
                    xp = xp + C[r + 1, j] * Xc[r]

                # interpolate kappa at the collocation point
                kappa_col = kappa_interp(k + tau[j])

                # append collocation equations (system dynamic)
                fj, qj = f_dyn(Xc[j - 1], Uk, kappa_col)
                g.append(mesh_point_separations[k] * fj - xp)
                lbg.append([0.0] * nx)
                ubg.append([0.0] * nx)

                # add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                # add contribution to quadrature function
                J = J + B[j] * qj * mesh_point_separations[k]

                # add contribution to scaling factor (for calculating lap time)
                sf_opt.append(B[j] * qj * mesh_point_separations[k])
                curv.append(kappa_col)
                
            # get middle curvature collocation point
            k_opt.append(curv[1])

            # calculate time step and used energy
            dt_opt.append(np.sum(sf_opt))
            motor_speed = self.veh["gear_ratio"]*(Xk[0]*v_s/re) # no slip ratio 
            ec_opt.append(motor_speed * Uk[1] * torque_drive_s * dt_opt[-1])


            # add new decision variables for the state at end of the collocation interval
            Xk = ca.MX.sym("X_" + str(k + 1), nx)
            w.append(Xk)
            lbw.append(
                [
                    v_min,
                    beta_min,
                    omega_z_min,
                    n_min,
                    xi_min
                ]
            )
            ubw.append(
                [
                    v_max,
                    beta_max,
                    omega_z_max,
                    n_max,
                    xi_max
                ]
            )
            w0.append(
                [
                    vx_guess,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ]
            )

            # add equality constraint
            g.append(Xk_end - Xk)
            lbg.append([0.0] * nx)
            ubg.append([0.0] * nx)

            # path constraint: f_drive * f_brake == 0 (no simultaneous operation of brake and accelerator pedal)
            g.append(Uk[1] * Uk[2])
            lbg.append([0.0])
            ubg.append([0.0])


            # get tire forces (used for enforcing no pitch and roll)
            f_x_flk, f_x_frk, f_x_rlk, f_x_rrk = f_fx(Xk, Uk)
            f_y_flk, f_y_frk, f_y_rlk, f_y_rrk = f_fy(Xk, Uk)

            f_xk = f_x_flk + f_x_frk + f_x_rlk + f_x_rrk
            f_yk = f_y_flk + f_y_frk + f_y_rlk + f_y_rrk

            f_z_flk, f_z_frk, f_z_rlk, f_z_rrk = f_fz(Xk, Uk)

            roll_relaxation = 1e-2
            pitch_relaxation = 1e-5

            # path constraint: longitudinal wheel load transfer assuming Ky = 0 (no pitch)
            # g.append(Uk[3] * gamma_x_s * (self.veh["lf"] + self.veh["lr"]) + self.veh["cg_height"] * f_xk)
            # lbg.append([-pitch_relaxation])
            # ubg.append([pitch_relaxation])
            # traction forces act at the road
            # gamma is a force, not a moment; k_R*gamma_y acts up (down) at front left and down (up) at front right 
            # thus we have 2 additive contributions of k_R*gamma_y to the moment, each with r = wf/2, for the front 
            # for pitch, we have 
            # TODO - our formulation here requires k_roll = 0.5; otherwise uneven gamma_y on front v.s. back contributes to Ky?
            # for pitch, we have 2 additive contributions of k_P*gamma_x on the left side, with r = lf and r = lr


            # path constraint: lateral wheel load transfer assuming Kx = 0 (no roll)
            g.append(Uk[3] * (k_roll*self.veh["wf"] + (1-k_roll)*self.veh["wr"]) + (self.veh["cg_height"] * f_yk) / gamma_y_s )
            lbg.append([-roll_relaxation])
            ubg.append([roll_relaxation])

            # get constant friction coefficient
            mue_fl = self.veh["mu"]
            mue_fr = self.veh["mu"]
            mue_rl = self.veh["mu"]
            mue_rr = self.veh["mu"]

            # path constraint: Kamm's Circle for each wheel
            # F_drive directly controls Fx now; this shouldn't be an issue
            KC_scale = 1
            g.append(KC_scale*((f_x_flk / (mue_fl * f_z_flk)) ** 2 + (f_y_flk / (mue_fl * f_z_flk)) ** 2))
            g.append(KC_scale*((f_x_frk / (mue_fr * f_z_frk)) ** 2 + (f_y_frk / (mue_fr * f_z_frk)) ** 2))
            g.append(KC_scale*((f_x_rlk / (mue_rl * f_z_rlk)) ** 2 + (f_y_rlk / (mue_rl * f_z_rlk)) ** 2))
            g.append(KC_scale*((f_x_rrk / (mue_rr * f_z_rrk)) ** 2 + (f_y_rrk / (mue_rr * f_z_rrk)) ** 2))
            lbg.append([0.0] * 4)
            ubg.append([1.0*KC_scale] * 4)
            
            
            ##################### Implementation of the motor curve
            

            # Power cap in kW; make this a control variable later
            power_cap = 60*1000

            # path constraint: limitied engine power
            g.append(Xk[0] * Uk[1]/re)
            lbg.append([-np.inf])
            ubg.append([power_cap / (torque_drive_s * v_s)])
            
            # # Determine the motor speed in rad/s
            # known_motor_speed = self.veh["gear_ratio"]*(Xk[0]*v_s/re)  # no slip ratio 
            # known_motor_speed = known_motor_speed * 30/np.pi # convert to rpm

            # # Interpolate our motor curve to get the maximum allowed motor torque for this speed
            # motor_speeds, motor_torques = self.motor.get_motor_curve(power_cap) #rpm, Nm
            
            # # Manually implement piecewise linear interpolation
            # motor_torque_max = ca.MX(0)

            # # Forward difference linear interpolator
            # for i in range(1, len(motor_speeds)):
            #     # Create conditions for the intervals
            #     lower_bound = motor_speeds[i-1]
            #     upper_bound = motor_speeds[i]
            #     slope = (motor_torques[i] - motor_torques[i-1]) / (upper_bound - lower_bound)
            #     intercept = motor_torques[i-1] - slope * lower_bound

            #     # Add conditionally interpolated torque to the sum
            #     motor_torque_max += ca.fmax(0, ca.fmin(1, (known_motor_speed - lower_bound) / (upper_bound - lower_bound))) * (slope * known_motor_speed + intercept)

            

            # Constrain our motor torque to be less than the maximum

            # g.append(motor_torque_max - Uk[1]*torque_drive_s)
            # lbg.append([0.0])
            # ubg.append([ca.inf])

            
            delta_p.append(Uk[0] * delta_s)
            F_p.append(Uk[1] * torque_drive_s + Uk[2] * f_brake_s)

            # append outputs
            # After we constrain Xk and Uk, add them to our control vectors
            x_opt.append(Xk * x_s)
            u_opt.append(Uk * u_s)


        # formulate differentiation matrix (for regularization)
        # diff_matrix = np.eye(N)
        # for i in range(N - 1):
        #     diff_matrix[i, i + 1] = -1.0
        # diff_matrix[N - 1, 0] = -1.0

        # regularization (delta)
        # delta_p = ca.vertcat(*delta_p)
        # Jp_delta = ca.mtimes(ca.MX(diff_matrix), delta_p)
        # Jp_delta = ca.dot(Jp_delta, Jp_delta)

        # regularization (f_drive + f_brake)
        # F_p = ca.vertcat(*F_p)
        # Jp_f = ca.mtimes(ca.MX(diff_matrix), F_p)
        # Jp_f = ca.dot(Jp_f, Jp_f)

        # formulate objective
        # r_delta = 3
        # r_F = 1e-9
        # J = J + r_F * Jp_f + r_delta * Jp_delta

        # concatenate NLP vectors to form a large but sparse matrix equation
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        w0 = np.concatenate(w0)

        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # concatenate output vectors
        x_opt = ca.vertcat(*x_opt)
        u_opt = ca.vertcat(*u_opt)
        dt_opt = ca.vertcat(*dt_opt)
        k_opt = ca.vertcat(*k_opt)
        # ax_opt = ca.vertcat(*ax_opt)
        # ay_opt = ca.vertcat(*ay_opt)
        ec_opt = ca.vertcat(*ec_opt)

        # ------------------------------------------------------------------------------------------------------------------
        # CREATE NLP SOLVER ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        """
        In this section, we use CasADi's NLP functionality to solve our direct collocation problem
        The NLP will find the optimal set of decision variables w, subject to a series of constraints g, such that
        we minimize the objective function J (which is the total laptime) 
        """

        # set up the nlp using CasADi's notation
        nlp = {"f": J/70, "x": w, "g": g}

        # solver options
        opts = {"expand": True, 
                "ipopt": {
                    "max_iter": 50,
                    "tol": 1e-6,
                    "linear_solver": "mumps",  
                    "hessian_approximation": "limited-memory",  # use L-BFGS instead of exact Hessian
                    "limited_memory_max_history": 50,
                    "mu_strategy": "adaptive",
                    "warm_start_init_point": "yes",
                }}

        # create solver instance using IPOPT
        # print("dx is:", dx)
        # print("dx type is:", type(dx))
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # solve the NLP; w0 is our initial guess for the optimal solution set w; lbw and ubw are our bounds on w
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # error print in case of failure
        if solver.stats()["return_status"] != "Solve_Succeeded":
            print("\033[1;91m" + "\nERROR: OPTIMIZATION DID NOT SUCCEED!" + "\033[0m")

        # ------------------------------------------------------------------------------------------------------------------
        # EXTRACT SOLUTION -------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        
        """
        In this section, we format the result of the solver into a convenient output
        """

        # helper function to extract solution for state variables, control variables, and times
        f_sol = ca.Function(
            "f_sol", [w], [x_opt, u_opt, dt_opt, k_opt, ec_opt], ["w"], ["x_opt", "u_opt", "dt_opt", "k_opt", "ec_opt"]
        )

        # extract solution
        x_opt, u_opt, dt_opt, k_opt, ec_opt = f_sol(sol["x"])

        # solution for state variables
        x_opt = np.reshape(x_opt, (N + 1, nx))

        # solution for control variables
        u_opt = np.reshape(u_opt, (N, nu))

        empty_row = np.full((1, u_opt.shape[1]), np.nan)  # Create an empty row filled with NaN
        u_opt = np.vstack((u_opt, empty_row))  # Add the empty row


        # solution for time
        t_opt = np.hstack((0.0, np.cumsum(dt_opt)))

        # solution for energy consumption (kWh)
        ec_opt_cum = np.hstack((0.0, np.cumsum(ec_opt))) / 3600.0 / 1000

        # Convert numpy arrays to pandas DataFrames, making them vertical vectors
        t_opt = np.reshape(t_opt, (-1, 1))  
        ec_opt_cum = np.reshape(ec_opt_cum, (-1, 1))
        
        # solution for curvature
        k_opt = np.array(k_opt)
        k_opt = np.append(k_opt[-1], k_opt)
        
        df0 = np.column_stack(
            (t_opt, k_opt, x_opt, u_opt, ec_opt_cum)
        )  
        df1 = pd.DataFrame(df0)  # reform the state as a dataframe
        
        header = [
            "time",
            "kappa",
            "v",
            "beta",
            "omega_z",
            "n",
            "xi",
            "delta",
            "torque_drive",
            "f_brake",
            "gamma_y",
            "energy"
        ]

        df1.columns = header

        output_name = self.track_file[:-5] + f'{segment_number}'
        save_output(df1, 'simple_bicycle_' + output_name + '.xlsx')

        return df1




class SweepWrapper:
    def __init__(self, tracks, param_sweep, parts, mesh=0.25):
        print("\033[1;92m" + "\nINITIALIZED SWEEPER" + "\033[0m")
        self.tracks = tracks
        self.param_sweep = param_sweep
        self.parts = parts
        self.mesh = mesh        # make a mesh variable for each track

    # calculate the number of points corresponding to event performance using rulebook
    def points_estimate(self, laptimes, energies): 
        # predict our efficiency factor and score from the average laptime and energy drain
        def calculate_efficiency_factor(avg_laptime, avg_energy_drain):
            # assume LapTotal min, yours, and CO2 are all the same (22)
            t_min = 67.667
            t_max = 98.118
            co2_min = 0.0518                            # avg kg CO2 per lap
            eff_factor_min = 0.0591
            eff_factor_max = 0.841

            
            # predict RIT's efficiency factor
            '''
            t_yours = 73.467
            e_yours = 4.903                             # kWh
            co2_yours = e_yours * 0.65
            co2_yours = co2_yours / 22                  # avg adjusted kg CO2 per lap
            RIT_factor = t_min/t_yours * co2_min / co2_yours
            RIT_score = 100 * (RIT_factor-eff_factor_min)/(eff_factor_max-eff_factor_min)
            
            print(RIT_factor)
            print(RIT_score)
            print(eff_factor_min)
            '''
            
            # predict our efficiency factor
            co2_yours = avg_energy_drain * 0.65          # avg adjusted kg CO2 per lap
            eff_factor = t_min/avg_laptime * co2_min / co2_yours


            #linear score approximator from eff_factor
            m = (100-81.3)/(0.841-0.243)
            Points = 89.9+m*(eff_factor-0.362)
            #print(Points)

            return eff_factor, Points
        

        # endurance time and energy are for the whole race (22 laps)
        # autoX and acceleration times are for 1 run

        reference_values = [1581.258,  # Minimum endurance time
                            46.776,    # Minimum autocross time
                            3.642,     # Minimum acceleration time
                            4.898]     # Minimum skidpad time


        # Reading reference values from ptsRef file
        minimum_endurance_time = reference_values[0]/22
        minimum_autoX_time = reference_values[1]
        minimum_acceleration_time = reference_values[2]
        minimum_skidpad_time = reference_values[3]
        
        max_endurance_time = 1.45*minimum_endurance_time
        max_autoX_time = 1.45*minimum_autoX_time
        max_accel_time = 1.5*minimum_acceleration_time
        max_skidpad_time = 1.25*minimum_skidpad_time

        # Calculate points for Endurance, Autocross, Skidpad, and Acceleration (from rulebook)
        points = {}
        # TODO assuming all laps are completed we should add 25 points. Is this correct?
        if 'endurance' in laptimes:
            points['endurance'] = 25 + min(250, 250*((max_endurance_time/laptimes['endurance'])-1)/((max_endurance_time/minimum_endurance_time)-1))

            if 'endurance' in energies:
                # hardcoded results from 2023
                CO2_min = 0.0518
                minimum_endurance_time = 67.667
                eff_factor_min = 0.059
                eff_factor_max = 0.841

                # estimate our efficiency factor from 2023 results 
                efficiency_factor, efficiency_score_2023 = calculate_efficiency_factor(laptimes['endurance'], energies['endurance'])
                efficiency_score_2024 = min(100, 100*(efficiency_factor-eff_factor_min)/(eff_factor_max-eff_factor_min))

                points['efficiency'] = efficiency_score_2024

        if 'autoX' in laptimes:
            points['autoX'] = min(125, 118.5*((max_autoX_time/laptimes['autoX'])-1)/((max_autoX_time/minimum_autoX_time)-1) + 6.5)

        if 'skidpad' in laptimes:
            points['skidpad'] = min(75, 71.5*(((max_skidpad_time/laptimes['skidpad'])**2 - 1)/((max_skidpad_time/minimum_skidpad_time)**2 - 1)) + 3.5)
        
        if 'accel' in laptimes:
            points['accel'] = min(100, 95.5*((max_accel_time/laptimes['accel'])-1)/((max_accel_time/minimum_acceleration_time)-1) + 4.5)


        return points


    # Estimates competition points gained by given vehicle
    def get_points(self, vehicle):
        # mesh = 0.25 #m

        # Define test track; TODO move this up in our structure
        basename = "track_files/"
        end_file = 'Michigan_2021_Endurance.xlsx'
        autoX_file = 'Michigan_2022_AutoX.xlsx'
        skidpad_file = 'Skidpad.xlsx'
        accel_file = 'Acceleration.xlsx'

        track_files = {'endurance': end_file, 
                       'autoX': autoX_file, 
                       'skidpad': skidpad_file, 
                       'accel': accel_file}
        
        laptimes = {}
        energies = {}

        # Instantiate and run solvers for each event
        for event, file in track_files.items():

            # If we're considering this event
            if event in self.tracks:
                # Create a new solver for the given track and vehicle
                solver = BicycleModel(basename+file, self.mesh, self.parts[event])
                solver.update_vehicle(vehicle)

                laptime, energy = solver.run(simple=True)
                laptimes[event] = laptime
                energies[event] = energy

        # Use the results of our sim to predict our competition points performance
        points = self.points_estimate(laptimes, energies)
        total = sum(points.values())
        points['total'] = total

        return laptimes, energies, points
        
    # Run get_points for several vehicles, combining the outputs into one massive df for output
    def sweep(self):
        '''
        What parameters do we use / need to sweep over for time targets?

        mass
        capacity (ignore for now)
        friction coefficients (ignore this for now; more tire stuff later)
        tire radius
        lift and drag coefficients
        front aero distribution / center of pressure location
        '''

        base_vehicle = Vehicle()

        # Create vehicle objects corresponding to all the parameters
        swept_vehicles = base_vehicle.sweep_parameters(self.param_sweep)    

        # Calculate get_points for each vehicle and summarize the results
        summary = []
        param_summary = []
        for i, vehicle in enumerate(swept_vehicles):
            vehicle_summary = {
                "Vehicle ID": i + 1,  # Assign a vehicle ID for reference
            }
            
            laptimes, energies, points = self.get_points(vehicle)
            
            # Log vehicle parameters corresponding to each ID
            param_summary.append({"Vehicle ID": i + 1, **vehicle.params})

            # Add lap times, energies, and points
            for track in self.tracks:
                vehicle_summary[f"Lap Time - {track}"] = laptimes[track]
                vehicle_summary[f"Energy - {track}"] = energies[track]
                vehicle_summary[f"Points - {track}"] = points[track]

            # Include the total points separately
            vehicle_summary["Total Points"] = points["total"]

            summary.append(vehicle_summary)


        # Export output summary
        param_df = pd.DataFrame(param_summary)
        summary_df = pd.DataFrame(summary)
        print(summary_df)
        save_output(summary_df, "sweep_results.xlsx", metadata=param_df)
        

def main_test():
    # Identify which events are relevant to our study    
    tracks = [
                 'endurance', 
                 'autoX', 
                 'skidpad', 
                 'accel'
            ]

    # Identify which parameters we want to sweep over
    param_sweep = {
                "M": [310],                                     # Mass in kg
                # "Cl": [-3, -2, -1, -0.3],                       # Coefficient of lift (- for downforce)
                # "Cd": [-3, -2, -1, -0.3],                       # Coefficient of drag (- for drag)
                # "grip_factor": [1.0] #, 0.85, 1],                  # Grip factor (crude estimate of driver "maximizing grip usage")
                # "re": [0.35],                                   # Effective wheel radius, meters
                # "mu": [0.75]#, 0.9]

    }

    # Divide each track into n segments to improve computation time
    # Adjust these based on your computer's abilities; higher number = faster, less accurate
    parts = {
                'endurance': 1, 
                'autoX': 6, 
                'skidpad': 1, 
                'accel': 1
            }

    sweeper = SweepWrapper(tracks, param_sweep, parts, 1.0)
    sweeper.sweep()



main_test()