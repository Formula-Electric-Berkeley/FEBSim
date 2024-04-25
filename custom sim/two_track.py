import numpy as np
import casadi as ca
import pandas as pd

import track as tr
import vehicle as veh

# Define constants
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


def opt_mintime():
    # ------------------------------------------------------------------------------------------------------------------
    # GET TRACK INFORMATION --------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get the curvature of the track centerline at each mesh point
    kappa_refline_cl = tr.curvatures

    # the total number of mesh points
    num_mesh_points = len(tr.curvatures)

    # optimization steps (0, 1, 2 ... end point)
    steps = [i for i in range(num_mesh_points)]
    N = steps[-1]

    # separation between each mesh point (in the future, this could be vary between points!)
    mesh_point_separations = np.ones(num_mesh_points) * tr.mesh_size

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
    """

    # define physical constants
    g = 9.81  # [m/s^2]
    mass = veh.M  # [kg]
    lf = veh.lf  # front wheelbase length (from COM to front axle)
    lr = veh.lr  # rear  wheelbase length
    L = lf + lr  # total wheelbase length
    wf = veh.wf  # front axle width (distance between centers of wheels)
    wr = veh.wr  # rear axle width

    Je = veh.Je  # engine moment of inertia TODO -> modify with real data
    Jw = veh.Jw  # wheel moment of inertia
    re = veh.re  # effective wheel radius
    rb = veh.rb  # effective brake radius
    Iz = veh.Iz  # vehicle moment of inertia about z axis

    # compute normal forces at each wheel, assumes wf roughly equals wr
    # this will be used by our tire model to compute grip forces at each wheel
    F_Nfr = (mass * g * lr) / (L * 2) + gamma_x - gamma_y
    F_Nfl = (mass * g * lr) / (L * 2) + gamma_x + gamma_y
    F_Nrr = (mass * g * lf) / (L * 2) - gamma_x - gamma_y
    F_Nrl = (mass * g * lf) / (L * 2) - gamma_x + gamma_y

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
    alpha_fr = -ca.arctan(
        vtfr / ca.fabs(vlfr)
    )  # TODO: old code used the smooth_abs helper function; replace if fucked
    alpha_fl = -ca.arctan(vtfl / ca.fabs(vlfl))
    alpha_rr = -ca.arctan(vtrr / ca.fabs(vlrr))
    alpha_rl = -ca.arctan(vtrl / ca.fabs(vlrl))

    # compute the slip ratios
    sigma_fr = (re * wfr - vlfr) / ca.fabs(vlfr)
    sigma_fl = (re * wfl - vlfl) / ca.fabs(vlfl)
    sigma_rr = (re * wrr - vlrr) / ca.fabs(vlrr)
    sigma_rl = (re * wrl - vlrl) / ca.fabs(vlrl)

    # compute wheel grip forces in the wheel frame
    Flfr, Ftfr = combined_slip_forces(
        sigma_fr, alpha_fr, F_Nfr
    )  # Format: F_longitudinal_front_right
    Flfl, Ftfl = combined_slip_forces(sigma_fl, alpha_fl, F_Nfl)
    Flrr, Ftrr = combined_slip_forces(sigma_rr, alpha_rr, F_Nrr)
    Flrl, Ftrl = combined_slip_forces(sigma_rl, alpha_rl, F_Nrl)

    # change wheel forces to body frame
    Fx_fr = Flfr * ca.cos(delta) - Ftfr * ca.sin(delta)
    Fy_fr = Flfr * ca.sin(delta) + Ftfr * ca.cos(delta)
    Fx_fl = Flfl * ca.cos(delta) - Ftfl * ca.sin(delta)
    Fy_fl = Flfl * ca.sin(delta) + Ftfl * ca.cos(delta)
    Fx_rr = Flrr
    Fy_rr = Ftrr
    Fx_rl = Flrl
    Fy_rl = Ftrl

    # drag forces
    Fx_aero = (
        -veh.drag_coeff * vx * ca.fabs(vx)
    )  # TODO add rolling resistance here later

    # compute the net force and torque on the vehicle
    Fx = (
        Fx_fr + Fx_fl + Fx_rl + Fx_rr + Fx_aero
    )  # net force in the x (longitudinal car direction)
    Fy = Fy_fr + Fy_fl + Fy_rr + Fy_rl  # net force in the y (lateral car direction)
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
        - Fx_aero * ca.cos(beta)
    )

    dbeta = sf * (
        -omega_z
        + (
            -(Flrl + Flrr) * ca.sin(beta)
            + (Flfl + Flfr) * ca.sin(delta - beta)
            + (Ftrl + Ftrr) * ca.cos(beta)
            + (Ftfl + Ftfr) * ca.cos(delta - beta)
            + Fx_aero * ca.sin(beta)
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

    # define our driving dynamics with a series of ordinary differential equations
    dx = (
        ca.vertcat(
            dv, dbeta, omegaz_dot, n_dot, xi_dot, wfr_dot, wfl_dot, wrl_dot, wrr_dot
        )
        / x_s
    )

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
    # TODO: explicitly constrain wfr... based on v at each collocation point to ensure slip ratio < 1

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
    # ec_opt = []

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

        # calculate time step
        dt_opt.append(sf_opt[0] + sf_opt[1] + sf_opt[2])

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
        g.append(Uk[3] * gamma_x_s * (veh.lf + veh.lr) + veh.cg_height * f_xk)
        lbg.append([0.0])
        ubg.append([0.0])

        # path constraint: lateral wheel load transfer assuming Kx = 0 (no roll)
        g.append(Uk[4] * gamma_y_s * (veh.wf + veh.wr) + veh.cg_height * f_yk)
        lbg.append([0.0])
        ubg.append([0.0])

        """
        Constrain v and torque according to the motor curve; interpolating spline
        What's stopping us from pushing max v? 

        We need to limit current/power draw also

        MAJOR TODO
        1. add all physical constants
        2. add motor torque curve
        3. add power limit / energy limit constraints (these will fix our motor curve)
        """

        # TODO: remaining issues
        # fix alternating between drive and brake
        # fix hard steering

        # limit rate of change of steering and drive commands (dynamic actuator constraints)
        delta_step = 4  # right now, we limit the rate of change of steering to be 1/4th the maximum value in a time-step
        drive_step = 1
        brake_step = 4

        if k > 0:
            # calculate the time-distance scale factor at the kth collocation point
            sf_k = (1 - kappa_interp(k) * Xk[3] * n_s) / (
                Xk[0] * v_s * np.cos(Xk[1] * beta_s + Xk[4] * xi_s)
            )

            # limit the rate of change *in time* between consecutive steering/drive/brake commands
            g.append(
                (Uk - w[1 + (d + 2) * (k - 1)]) / (mesh_point_separations[k - 1] * sf_k)
            )

            # we impose no bounds on how fast drive and brake commands can go to 0
            lbg.append(
                [
                    delta_min / delta_step,
                    -np.inf,
                    f_brake_min / brake_step,
                    -np.inf,
                    -np.inf,
                ]
            )
            ubg.append(
                [
                    delta_max / delta_step,
                    f_drive_max / drive_step,
                    np.inf,
                    np.inf,
                    np.inf,
                ]
            )

                        # we impose no bounds on how fast drive and brake commands can go to 0
                        lbg.append([delta_min / delta_step, -np.inf, f_brake_min / brake_step, -np.inf, -np.inf])
                        ubg.append([delta_max / delta_step, f_drive_max / drive_step, np.inf, np.inf, np.inf])

                
                # append controls (for regularization)
                #delta_p.append(Uk[0] * delta_s)
                #F_p.append(Uk[1] * torque_drive_s / 10000.0 + Uk[2] * f_brake_s / 10000.0)

                # append outputs, renormalized to their proper size



                #TODO start here!!!!!!!

                '''
                # do both of the back wheels need to have the same angular velocity 
                # there is some difference in slip, but this could be because the velocity projection is different
                # motor and brake torques are the same, but if grip torques are ever different (i.e. different velocity projection or slip angle), then ws would differ
                # UNLESS we actively constrain that

                torque_max_motor = f(angular speed of back axle, power cap)
                torque_commanded - torque_max_motor <= 0

                '''


        # ENERGISTICS TODO

        """
        Based on our commanded torque and velocity, we can find where on the motor efficiency curve we are
        This can determine our regen efficiency
        We can also add a variable/functional power cap to limit rate of regen

        Energy consumption can come from 
        """

        x_opt.append(Xk * x_s)
        u_opt.append(Uk * u_s)

    # boundary constraint: start states = final states (only needed for a closed track)
    # g.append(w[0] - Xk)
    # lbg.append([0.0] * nx)
    # ubg.append([0.0] * nx)

    # Regularization ---> ensures smoothing of control variables so we don't oscillate
    """
    # formulate differentiation matrix (for regularization)
    diff_matrix = np.eye(N)
    for i in range(N - 1):
            diff_matrix[i, i + 1] = -1.0
    diff_matrix[N - 1, 0] = -1.0

    # regularization (delta)
    delta_p = ca.vertcat(*delta_p)
    Jp_delta = ca.mtimes(ca.MX(diff_matrix), delta_p)
    Jp_delta = ca.dot(Jp_delta, Jp_delta)

    # regularization (f_drive + f_brake)
    F_p = ca.vertcat(*F_p)
    Jp_f = ca.mtimes(ca.MX(diff_matrix), F_p)
    Jp_f = ca.dot(Jp_f, Jp_f)

    #Penalties (put somwhere else later)
    rf = 0.01
    rdelta = 10

    # formulate objective

    #Smoothing helps remove oscillations and convergence problems, but leads to an unphysical increase in lap time
    #Adding "pentalties" to the objective function ensures we don't over-smooth
    J = J #+ rf * Jp_f + rdelta * Jp_delta
    """

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
    # ax_opt = ca.vertcat(*ax_opt)
    # ay_opt = ca.vertcat(*ay_opt)
    # ec_opt = ca.vertcat(*ec_opt)

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
        "f_sol", [w], [x_opt, u_opt, dt_opt], ["w"], ["x_opt", "u_opt", "dt_opt"]
    )

    # extract solution
    x_opt, u_opt, dt_opt = f_sol(sol["x"])

    # solution for state variables
    x_opt = np.reshape(x_opt, (N + 1, nx))

    # solution for control variables
    u_opt = np.reshape(u_opt, (N, nu))

    # solution for time
    t_opt = np.hstack((0.0, np.cumsum(dt_opt)))

    print("Optimal laptime is: ", t_opt)

    # Convert numpy arrays to pandas DataFrames
    t_opt = np.reshape(t_opt, (-1, 1))  # make t_opt a vertical vector
    df0 = np.concatenate(
        (t_opt, x_opt), axis=1
    )  # combine t_opt with our state solution
    df1 = pd.DataFrame(df0)  # reform the state as a dataframe
    df2 = pd.DataFrame(u_opt)
    df1 = pd.concat([df1, df2], axis=1)  # combine our state and control solutions

    header = [
        "time",
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
    ]

    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter("arrays13.xlsx", engine="xlsxwriter")

    # Write the dataframe to a excel sheet
    df1.to_excel(writer, sheet_name="Sheet1", index=False, header=header)

    # Close the Pandas Excel writer and output the excel file
    writer.close()


opt_mintime()
