'''
More advanced model for laptime simulation. 
'''
import numpy as np
import casadi as ca
import pandas as pd
from track_for_bicycle import Track
import vehicle as veh
import powertrain_model
from scipy.interpolate import interp1d
import os

# Define tire constants
mu = 0.75  # coefficient of friction for tires
# pacejka model
Bx = 16
Cx = 1.58
Ex = 0.1

By = 13
Cy = 1.45
Ey = -0.8


# TODO: replace with Nick's tire model later
# This model can handle slip ratios outside [-1, 1], but we shouldn't get there
def combined_slip_forces(s, a, Fn):
    # Get forces pacejka model
    Fx0 = mu * Fn * ca.sin(Cx * ca.arctan((1 - Ex) * Bx * s + Ex * ca.arctan(Bx * s)))
    Fy0 = mu * Fn * ca.sin(Cy * ca.arctan((1 - Ey) * By * a + Ey * ca.arctan(By * a)))

    Exa = -Fn / 10000
    Dxa = 1
    Cxa = 1
    Bxa = 13 * ca.cos(ca.arctan(9.7 * s))
    Gxa = Dxa * ca.cos(Cxa * ca.arctan(Bxa * a - Exa * (Bxa * a - ca.arctan(Bxa * a))))

    Eys = 0.3
    Dys = 1
    Cys = 1
    Bys = 10.62 * ca.cos(ca.arctan(7.8 * a))
    Gys = Dys * ca.cos(Cys * ca.arctan(Bys * s - Eys * (Bys * s - ca.arctan(Bys * s))))

    Fx = Fx0 * Gxa
    Fy = Fy0 * Gys

    return Fx, Fy



class Vehicle:
    def __init__(self):
        # Initialize the vehicle with 30kW SN3 and allow for altered parameters
        self = veh
    

class BicycleModel:
    def __init__(self, track_file, mesh_size, track_parts):
        # Initialize the track and vehicle by loading the relevant files
        self.initialize_track(track_file, mesh_size, track_parts)
        self.initialize_vehicle()
        self.mesh_size = mesh_size

    def initialize_track(self, trackfile, mesh_size, parts):
        self.tr = Track(trackfile, mesh_size)

        # Break the track into parts splines of curvature
        self.parts = self.tr.split(parts)

    def initialize_vehicle(self, veh):
        self.motor = powertrain_model.motor()
        self.diff = powertrain_model.drexler_differential()


    def run(self):
        dfs = []
        for i, part in enumerate(self.parts):
            print('Running part {}: '.format(i))
            dfs.append(self.opt_mintime(part, self.mesh_size))

        combined_df = pd.concat(dfs, axis=0, ignore_index=True)

        self.save_output(combined_df, 'combined_output.xlsx')

        laptime = sum(df["time"].iloc[-1] for df in dfs if not df.empty)

        energy = sum(df["energy"].iloc[-1] for df in dfs if not df.empty)

        print('Total laptime is {:.2f} seconds with an energy consumption of {:.2f} kWh'.format(laptime, energy))

        return laptime, energy

        


    def save_output(self, data, filename, directory='sims_logs/'):
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
        

        # Write the dataframe to a excel sheet
        data.to_excel(file_path, sheet_name="Sheet1", index=False)

    def opt_mintime(self, curvatures, mesh_size):
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
        nx = 9

        # velocity [m/s]
        v_n = ca.SX.sym(
            "v_n"
        )  # we give CasADi's solver normalized variables to improve convergence
        v_s = 30  # scale the normalized vector
        v = v_s * v_n

        # sideslip angle [rad/s]
        beta_n = ca.SX.sym("beta_n")
        beta_s = 0.5
        beta = beta_s * beta_n

        # yaw rate [rad/s]
        omega_z_n = ca.SX.sym("omega_z_n")
        omega_z_s = 1
        omega_z = omega_z_s * omega_z_n

        # lateral distance to reference line (positive = left) [m]
        n_n = ca.SX.sym("n_n")
        n_s = 5.0
        n = n_s * n_n

        # relative angle to tangent on reference line [rad]
        xi_n = ca.SX.sym("xi_n")
        xi_s = 1.0
        xi = xi_s * xi_n

        # wheel angular velocities [rad/s]
        wfr_n = ca.SX.sym("wfr_n")
        wfl_n = ca.SX.sym("wfl_n")
        wrl_n = ca.SX.sym("wrl_n")
        wrr_n = ca.SX.sym("wrr_n")

        wheel_scale = 300
        wfr_s = wheel_scale
        wfl_s = wheel_scale
        wrr_s = wheel_scale
        wrl_s = wheel_scale

        wfr = wfr_s * wfr_n
        wfl = wfl_s * wfl_n
        wrr = wrr_s * wrr_n
        wrl = wrl_s * wrl_n

        x_s = np.array([v_s, beta_s, omega_z_s, n_s, xi_s, wfr_s, wfl_s, wrl_s, wrr_s])

        # vertical vector for the state variables
        x = ca.vertcat(v_n, beta_n, omega_z_n, n_n, xi_n, wfr_n, wfl_n, wrl_n, wrr_n)

        # ------------------------------------------------------------------------------------------------------------------
        # CONTROL VARIABLES ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # number of control variables
        nu = 5

        # steer angle [rad]
        delta_n = ca.SX.sym("delta_n")
        delta_s = 0.5
        delta = delta_s * delta_n

        # positive drive torque from the motor [N-m]
        torque_drive_n = ca.SX.sym("f_drive_n")
        torque_drive_s = 6000.0
        torque_drive = torque_drive_s * torque_drive_n

        # negative longitudinal force (brake) [N]
        f_brake_n = ca.SX.sym("f_brake_n")
        f_brake_s = 20000.0
        f_brake = f_brake_s * f_brake_n

        # longitudinal wheel load transfer [N]
        gamma_x_n = ca.SX.sym("gamma_x_n")
        gamma_x_s = 5000.0
        gamma_x = gamma_x_s * gamma_x_n

        # lateral wheel load transfer [N]
        gamma_y_n = ca.SX.sym("gamma_y_n")
        gamma_y_s = 5000.0
        gamma_y = gamma_y_s * gamma_y_n

        # curvature of reference line [rad/m] (allows CasADi to interpret kappa in our math below)
        kappa = ca.SX.sym("kappa")

        # scaling factors for control variables
        u_s = np.array([delta_s, torque_drive_s, f_brake_s, gamma_x_s, gamma_y_s])

        # put all controls together
        u = ca.vertcat(delta_n, torque_drive_n, f_brake_n, gamma_x_n, gamma_y_n)

        # ------------------------------------------------------------------------------------------------------------------
        # MODEL PHYSICS ----------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        """
        In this section, we define the physical model equations that constrain our system
        The physical equations here ultimately define how our state variables evolve over time (and thus over the lap)

        TODO: ensure the physics is correct -> use Ackermann geometry
        """


        # TODO; this is where we should change our vehicle

        # define physical constants and vehicle parameters; pass these in in the future
        g = 9.81  # [m/s^2]
        mass = veh.M    # [kg]
        lf = veh.lf     # front wheelbase length (from COM to front axle)
        lr = veh.lr     # rear  wheelbase length
        L = lf + lr     # total wheelbase length
        wf = veh.wf     # front axle width (distance between centers of wheels)
        wr = veh.wr     # rear axle width

        Je = veh.Je     # engine moment of inertia TODO -> modify with real data
        Jw = veh.Jw     # wheel moment of inertia
        re = veh.re     # effective wheel radius
        rb = veh.rb     # effective brake radius
        Iz = veh.Iz     # vehicle moment of inertia about z axis

        # Aero stuff
        air_density = veh.rho
        frontal_area = veh.A
        Cl = veh.Cl      # NEGATIVE Cl means downforce
        Cd = veh.Cd      # negative Cd means drag

        drag_coeff = 0.5*air_density*Cl*frontal_area
        df_coeff = - 0.5*air_density*Cd*frontal_area    # "Positive force" = down

        # TODO; WIP implementation 9/30/24
        # Compute aerodynamic and drag forces
        # Assumes df shared equally bw all wheels and that front area and Cl/Cd do not change w angle of attack
        # Also assumes 0.5 COP right now; this can change later
        Fx_aero_drag = drag_coeff * v * v
        Fz_aero_down = df_coeff * v * v          

        # TODO; fix roll and dynamic load transfer calculations
        '''
        f_xroll_fl = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_rear"] / veh["wheelbase"]
        f_xroll_fr = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_rear"] / veh["wheelbase"]
        f_xroll_rl = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_front"] / veh["wheelbase"]
        f_xroll_rr = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_front"] / veh["wheelbase"]
        f_xroll = tire["c_roll"] * mass * g

        f_zdyn_fl = (-0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll)
                    - veh["k_roll"] * gamma_y)
        f_zdyn_fr = (-0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll)
                    + veh["k_roll"] * gamma_y)
        f_zdyn_rl = (0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll)
                    - (1.0 - veh["k_roll"]) * gamma_y)
        f_zdyn_rr = (0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll)
                    + (1.0 - veh["k_roll"]) * gamma_y)
        
        '''
        

        # compute normal forces at each wheel, assumes wf roughly equals wr
        # this will be used by our tire model to compute grip forces at each wheel
        F_Nfr_weight = (mass * g * lr) / (L * 2) + gamma_x - gamma_y
        F_Nfl_weight = (mass * g * lr) / (L * 2) + gamma_x + gamma_y
        F_Nrr_weight = (mass * g * lf) / (L * 2) - gamma_x - gamma_y
        F_Nrl_weight = (mass * g * lf) / (L * 2) - gamma_x + gamma_y

        F_Nfr = F_Nfr_weight + Fz_aero_down*0.25
        F_Nfl = F_Nfl_weight + Fz_aero_down*0.25
        F_Nrr = F_Nrr_weight + Fz_aero_down*0.25
        F_Nrl = F_Nrl_weight + Fz_aero_down*0.25

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
        alpha_fr = -ca.arctan(vtfr / ca.fabs(vlfr))  
        alpha_fl = -ca.arctan(vtfl / ca.fabs(vlfl))
        alpha_rr = -ca.arctan(vtrr / ca.fabs(vlrr))
        alpha_rl = -ca.arctan(vtrl / ca.fabs(vlrl))

        # compute the slip ratios
        epsilon = 0.05 # smoothing parameter
        sigma_fr = (re * wfr - vlfr) / ca.fabs(vlfr)
        sigma_fl = (re * wfl - vlfl) / ca.fabs(vlfl)
        sigma_rr = (re * wrr - vlrr) / ca.fabs(vlrr)
        sigma_rl = (re * wrl - vlrl) / ca.fabs(vlrl)

        # compute wheel grip forces in the wheel frame
        Flfr, Ftfr = combined_slip_forces(0.0, alpha_fr, F_Nfr)  
        Flfl, Ftfl = combined_slip_forces(0.0, alpha_fl, F_Nfl)
        Flrr, Ftrr = combined_slip_forces(0.0, alpha_rr, F_Nrr)
        Flrl, Ftrl = combined_slip_forces(0.0, alpha_rl, F_Nrl)

        # change wheel forces to body frame
        Fx_fr = Flfr * ca.cos(delta) - Ftfr * ca.sin(delta)
        Fy_fr = Flfr * ca.sin(delta) + Ftfr * ca.cos(delta)
        Fx_fl = Flfl * ca.cos(delta) - Ftfl * ca.sin(delta)
        Fy_fl = Flfl * ca.sin(delta) + Ftfl * ca.cos(delta)
        Fx_rr = Flrr
        Fy_rr = Ftrr
        Fx_rl = Flrl    # Driven wheels are oriented parallel to the car in RWD
        Fy_rl = Ftrl

        
        # compute the net force and torque on the vehicle
        Fx = (
            Fx_fr + Fx_fl + Fx_rl + Fx_rr + Fx_aero_drag
        )  # net force in the x (longitudinal car direction)
        Fy = Fy_fr + Fy_fl + Fy_rr + Fy_rl  # net force in the y (lateral car direction)
        
        
        # TODO: fix THIS ASAP
        Kz = (
            lf * (Fy_fr + Fy_fl)
            - lr * (Fy_rr + Fy_rl)
            + wf / 2 * (Fx_fr - Fx_fl)
            + wr / 2 * (Fx_rr - Fx_rl)
        )  # net torque about z axis

        # distribute commanded brake force to the wheels based on our vehicle's brake force distribution
        F_brake_fr = veh.brake_fr * f_brake  # brake force of front right wheel
        F_brake_fl = veh.brake_fl * f_brake
        F_brake_rl = veh.brake_rl * f_brake
        F_brake_rr = veh.brake_rr * f_brake

        # calculate the effective torque experienced by each wheel due to braking
        # this partly determines the rate of change of the angular velocity of each wheel, which in turn is used to compute the slip ratio
        torque_brake_fr = rb * F_brake_fr
        torque_brake_fl = rb * F_brake_fl
        torque_brake_rr = rb * F_brake_rr
        torque_brake_rl = rb * F_brake_rl


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
            (Flrl + Flrr) * ca.cos(beta)
            + (Flfl + Flfr) * ca.cos(delta - beta)
            + (Ftrl + Ftrr) * ca.sin(beta)
            - (Ftfl + Ftfr) * ca.sin(delta - beta)
            - Fx_aero_drag * ca.cos(beta)
        )

        dbeta = sf * (
            -omega_z
            + (
                -(Flrl + Flrr) * ca.sin(beta)
                + (Flfl + Flfr) * ca.sin(delta - beta)
                + (Ftrl + Ftrr) * ca.cos(beta)
                + (Ftfl + Ftfr) * ca.cos(delta - beta)
                + Fx_aero_drag * ca.sin(beta)
            )
            / (mass * v)
        )

        # angular acceleration about the z axis
        omegaz_dot = sf * Kz / Iz

        xi_dot = sf * omega_z - kappa
        n_dot = sf * v * ca.sin(xi + beta)

        # compute the rates of change of the wheel angular velocities (wheel speeds) based on the brake torque, drive torque, and grip forces
        wfr_dot = sf * (-Flfr * re + torque_brake_fr) / Jw
        wfl_dot = sf * (-Flfl * re + torque_brake_fl) / Jw

        # For the driven wheels, 

        wrr_dot = (
            sf
            * (-Flrr * re + torque_brake_rr + torque_drive / 2)
            / (Jw + Je * veh.ratio_final / 2)
        )  # gear ratio
        wrl_dot = (
            sf
            * (-Flrl * re + torque_brake_rl + torque_drive / 2)
            / (Jw + Je * veh.ratio_final / 2)
        )

        # If wrr_dot and wrl_dot are positive, we're accelerating. Otherwise, we're decelerating



        # define our driving dynamics with a series of ordinary differential equations
        dx = (
            ca.vertcat(
                dv, dbeta, omegaz_dot, n_dot, xi_dot, wfr_dot, wfl_dot, wrl_dot, wrr_dot
            )
            / x_s
        )


        print("Dynamics Defined")
        # ------------------------------------------------------------------------------------------------------------------
        # CONTROL BOUNDARIES -----------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        """
        In the next two sections, we set the min and max values for each of the control and state variables
        The model will ensure these inequalities are satisfied at every collocation point
        Making sure these boundaries are physical (not too broad, not too narrow) is *crucial* for accurate convergence 
        Because we give CasADi normalized vectors, we must normalize these bounds by dividing by each variable's scale x_s
        """

        delta_min = -veh.delta_max / delta_s  # min. steer angle [rad]
        delta_max = veh.delta_max / delta_s  # max. steer angle [rad]

        f_drive_min = 0.0  # min. longitudinal drive TORQUE [Nm]
        f_drive_max = veh.drive_max / torque_drive_s  # max. longitudinal drive torque [Nm]

        # the value of our brake force is always negative
        f_brake_min = -veh.brake_max / f_brake_s  # min. longitudinal brake force [N]
        f_brake_max = 0.0  # max. longitudinal brake force [N]

        # The load transfer forces are unbounded; later we constrain them by enforcing no roll / pitch
        gamma_x_min = -np.inf  # min. longitudinal wheel load transfer [N]
        gamma_x_max = np.inf  # max. longitudinal wheel load transfer [N]

        gamma_y_min = -np.inf  # min. lateral wheel load transfer [N]
        gamma_y_max = np.inf  # max. lateral wheel load transfer [N]

        # ------------------------------------------------------------------------------------------------------------------
        # STATE BOUNDARIES -------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        v_min = 0.0 / v_s  # min. velocity [m/s]
        v_max = veh.max_velocity / v_s  # max. velocity [m/s]

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
        n_min = -4 / n_s  # min lateral distance from reference line [m]
        n_max = 4 / n_s

        # for simplicitly, wheels cannot spin backwards, and we cannot have slip ratios > 1
        wheelspeed_min = 0.0
        wheelspeed_max = 2 * veh.max_velocity / veh.re / wheel_scale

        # ------------------------------------------------------------------------------------------------------------------
        # INITIAL GUESS FOR DECISION VARIABLES -----------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        vx_guess = 20.0 / v_s

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
            [Fx_fl, Fx_fr, Fx_rl, Fx_rr],
            ["x", "u"],
            ["Fxfl", "Fxfr", "Fxrl", "Fxrr"],
        )
        # lateral tire forces in the vehicle frame [N]
        f_fy = ca.Function(
            "f_fy",
            [x, u],
            [Fy_fl, Fy_fr, Fy_rl, Fy_rr],
            ["x", "u"],
            ["Fyfl", "Fyfr", "Fyrl", "Fyrr"],
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
        # delta_p = []
        # F_p = []

        # boundary constraint: set initial conditions for the state variables
        Xk = ca.MX.sym("X0", nx)
        w.append(Xk)
        lbw.append(
            [
                v_min,
                0.0,
                0.0,
                n_min,
                0.0,
                wheelspeed_min,
                wheelspeed_min,
                wheelspeed_min,
                wheelspeed_min,
            ]
        )
        ubw.append(
            [
                v_max,
                0.0,
                0.0,
                n_max,
                0.0,
                wheelspeed_max,
                wheelspeed_max,
                wheelspeed_max,
                wheelspeed_max,
            ]
        )
        w0.append(
            [
                v_max,
                0.0,
                0.0,
                0.0,
                0.0,
                v_max / veh.re,
                v_max / veh.re,
                v_max / veh.re,
                v_max / veh.re,
            ]
        )
        x_opt.append(Xk * x_s)

        # loop along the racetrack and formulate path constraints & system dynamics
        for k in range(N):
            # add decision variables for the control
            Uk = ca.MX.sym("U_" + str(k), nu)
            w.append(Uk)
            lbw.append([delta_min, f_drive_min, f_brake_min, gamma_x_min, gamma_y_min])
            ubw.append([delta_max, f_drive_max, f_brake_max, gamma_x_max, gamma_y_max])
            w0.append([0.0] * nu)

            # add decision variables for the state at collocation points
            Xc = []
            for j in range(d):
                Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), nx)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append([-np.inf] * nx)
                ubw.append([np.inf] * nx)
                w0.append(
                    [
                        vx_guess,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        vx_guess / veh.re,
                        vx_guess / veh.re,
                        vx_guess / veh.re,
                        vx_guess / veh.re,
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
            dt_opt.append(sf_opt[0] + sf_opt[1] + sf_opt[2])
            motor_speed = (Xk[7] + Xk[8]) / (2*veh.gear_ratio)
            ec_opt.append(motor_speed * wheel_scale * Uk[1] * torque_drive_s * dt_opt[-1])


            # add new decision variables for the state at end of the collocation interval
            Xk = ca.MX.sym("X_" + str(k + 1), nx)
            w.append(Xk)
            lbw.append(
                [
                    v_min,
                    beta_min,
                    omega_z_min,
                    n_min,
                    xi_min,
                    wheelspeed_min,
                    wheelspeed_min,
                    wheelspeed_min,
                    wheelspeed_min,
                ]
            )
            ubw.append(
                [
                    v_max,
                    beta_max,
                    omega_z_max,
                    n_max,
                    xi_max,
                    wheelspeed_max,
                    wheelspeed_max,
                    wheelspeed_max,
                    wheelspeed_max,
                ]
            )
            w0.append(
                [
                    vx_guess,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    vx_guess / veh.re,
                    vx_guess / veh.re,
                    vx_guess / veh.re,
                    vx_guess / veh.re,
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

            # path constraint: longitudinal wheel load transfer assuming Ky = 0 (no pitch)
            roll_relaxation = 1e-7
            pitch_relaxation = 1e-7
            g.append(Uk[3] * gamma_x_s * (veh.lf + veh.lr) + veh.cg_height * f_xk)
            lbg.append([-pitch_relaxation])
            ubg.append([pitch_relaxation])

            # path constraint: lateral wheel load transfer assuming Kx = 0 (no roll)
            g.append(Uk[4] * gamma_y_s * (veh.wf + veh.wr) + veh.cg_height * f_yk)
            lbg.append([-roll_relaxation])
            ubg.append([roll_relaxation])


            # CHANGES: regularization, relaxed roll and pitch; added final constraint
            
            
            ##################### Implementation of the motor curve
            

            # Power cap in kW; make this a control variable later
            power_cap = 60
            
            # Determine the motor speed in rad/s
            known_motor_speed = (Xk[7] + Xk[8]) / (2*veh.gear_ratio)
            known_motor_speed = known_motor_speed * 30/np.pi # convert to rpm

            # Interpolate our motor curve to get the maximum allowed motor torque for this speed
            motor_speeds, motor_torques = self.motor.get_motor_curve(power_cap) #rpm, Nm
            
            # Manually implement piecewise linear interpolation
            motor_torque_max = ca.MX(0)

            # Forward difference linear interpolator
            for i in range(1, len(motor_speeds)):
                # Create conditions for the intervals
                lower_bound = motor_speeds[i-1]
                upper_bound = motor_speeds[i]
                slope = (motor_torques[i] - motor_torques[i-1]) / (upper_bound - lower_bound)
                intercept = motor_torques[i-1] - slope * lower_bound

                # Add conditionally interpolated torque to the sum
                motor_torque_max += ca.fmax(0, ca.fmin(1, (known_motor_speed - lower_bound) / (upper_bound - lower_bound))) * (slope * known_motor_speed + intercept)

            

            # Constrain our motor torque to be less than the maximum

            g.append(motor_torque_max - Uk[1])
            lbg.append([0.0])
            ubg.append([ca.inf])

            
                    
            """
            Remaining major to-dos
            1. Implement LSD differential
                - Implement the motor curve 
                - constrain motor torque based on motor speed
            2. Find the motor efficiency based on the motor torque and speed
            3. Optimize code

            Based on our commanded torque and velocity, we can find where on the motor efficiency curve we are
            This can determine our regen efficiency
            We can also add a variable/functional power cap to limit rate of regen

            """

            # delta_p.append(Uk[0] * delta_s)
            # F_p.append(Uk[1] * torque_drive_s / 10000.0 + Uk[2] * f_brake_s / 10000.0)

            # append outputs
            # After we constrain Xk and Uk, add them to our control vectors
            x_opt.append(Xk * x_s)
            u_opt.append(Uk * u_s)

        # boundary constraint: start states = final states (only needed for a closed track)
        # g.append(w[0] - Xk)
        # lbg.append([0.0] * nx)
        # ubg.append([0.0] * nx)

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

        # r_delta = 10 
        # r_F = 0.5 *10**-8
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
        nlp = {"f": J, "x": w, "g": g}

        # solver options
        opts = {"expand": True, "ipopt.max_iter": 20000, "ipopt.tol": 1e-7}

        # create solver instance using IPOPT
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # solve the NLP; w0 is our initial guess for the optimal solution set w; lbw and ubw are our bounds on w
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # error print in case of failure
        if solver.stats()["return_status"] != "Solve_Succeeded":
            print("\033[91m" + "ERROR: Optimization did not succeed!" + "\033[0m")

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

        # solution for energy consumption
        ec_opt_cum = np.hstack((0.0, np.cumsum(ec_opt))) / 3600.0

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
            "wfr",
            "wfl",
            "wbl",
            "wbr",
            "delta",
            "torque_drive",
            "f_brake",
            "gamma_x",
            "gamma_y",
            "energy"
        ]

        df1.columns = header

        self.save_output(df1, 'bicycle_out.xlsx')

        return df1
        

# estimate the number of points we will get with the car
def points_estimate(laptimes, energies): 
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
                        4.898]    # Minimum skidpad time


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
    points['endurance'] = 25 + min(250, 250*((max_endurance_time/laptimes['endurance'])-1)/((max_endurance_time/minimum_endurance_time)-1))
    points['autoX'] = min(125, 118.5*((max_autoX_time/laptimes['autoX'])-1)/((max_autoX_time/minimum_autoX_time)-1) + 6.5)
    points['skidpad'] = min(75, 71.5*(((max_skidpad_time/laptimes['skidpad'])**2 - 1)/((max_skidpad_time/minimum_skidpad_time)**2 - 1)) + 3.5)
    points['accel'] = min(100, 95.5*((max_accel_time/laptimes['accel'])-1)/((max_accel_time/minimum_acceleration_time)-1) + 4.5)


    # hardcoded results from 2023
    CO2_min = 0.0518
    minimum_endurance_time = 67.667
    eff_factor_min = 0.059
    eff_factor_max = 0.841

    # estimate our efficiency factor from 2023 results 
    efficiency_factor, efficiency_score_2023 = calculate_efficiency_factor(laptimes['endurance'], energies['endurance'])
    efficiency_score_2024 = min(100, 100*(efficiency_factor-eff_factor_min)/(eff_factor_max-eff_factor_min))

    points['efficiency'] = efficiency_score_2024

    return points

def run_test():
    # Define test track
    basename = "track_files/"
    file = "test_track.xlsx"
    # file = 'Michigan_2022_AutoX.xlsx'
    filename = basename + file
    mesh = 0.25 #m
    parts = 1

    # Instantiate and run solver
    solver = BicycleModel(filename, mesh, parts)
    solver.run()


def get_points():
    mesh = 0.25 #m

    # Define test track
    basename = "track_files/"
    end_file = 'Michigan_2021_Endurance.xlsx'
    autoX_file = 'Michigan_2022_AutoX.xlsx'
    skidpad_file = 'Skidpad.xlsx'
    accel_file = 'Acceleration.xlsx'

    tracks = ['endurance', 'autoX', 'skidpad', 'accel']
    track_files = [end_file, autoX_file, skidpad_file, accel_file]
    laptimes = {}
    energies = {}

    # Instantiate and run solvers for each event
    for idx, file in enumerate(track_files):
        # Partition endurance and autoX four times to ease convergence
        # TODO; WIP; these may need to be tuned more
        if idx == 3 or idx == 2:
            parts = 1
        else:
            parts = 5

        solver = BicycleModel(file, mesh, parts)
        laptime, energy = solver.run()
        laptimes[tracks[idx]] = laptime
        energies[tracks[idx]] = energy

    # Use the results of our sim to predict our competition points performance
    points = points_estimate(laptimes, energies)
    total = sum(points.values())

    header = [
            "endurance_time",
            "endurance_energy",
            "endurance_points",
            "efficiency_points",
            "autoX_time",
            "autoX_energy",
            "autoX_points",
            "skidpad_time",
            "skidpad_energy",
            "skidpad_points",
            "accel_time",
            "accel_energy",
            "accel_points",
            "total_points"
    ]

    results = pd.DataFrame(columns=header)

    # Populate our output df
    

def sweep_vehicles():
    '''
    What parameters do we use / need to sweep over for time targets?

    mass
    capacity (ignore for now)
    friction coefficients (ignore this for now; more tire stuff later)
    tire radius
    lift and drag coefficients
    front aero distribution / center of pressure location
    '''

# run_test()