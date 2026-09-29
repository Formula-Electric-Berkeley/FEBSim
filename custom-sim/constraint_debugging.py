import os
import sys
import time
import numpy as np
import casadi as ca
import track as tr
import vehicle as veh
import pandas as pd

import powertrain_model
from scipy.interpolate import interp1d


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



def opt_mintime():
    """
    Created by:
    Fabian Christ, Thomas Herrmann, Francesco Passigato

    Documentation:
    The minimum lap time problem is described as an optimal control problem, converted to a nonlinear program using
    direct orthogonal Gauss-Legendre collocation and then solved by the interior-point method IPOPT. Reduced computing
    times are achieved using a curvilinear abscissa approach for track description, algorithmic differentiation using
    the software framework CasADi, and a smoothing of the track input data by approximate spline regression. The
    vehicles behavior is approximated as a double track model with quasi-steady state tire load simplification and
    nonlinear tire model.

    Please refer to our paper for further information:
    Christ, Wischnewski, Heilmeier, Lohmann
    Time-Optimal Trajectory Planning for a Race Car Considering Variable Tire-Road Friction Coefficients
    """


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

    # degree of interpolating polynomial
    d = 4

    # legendre collocation points
    tau = np.append(0, ca.collocation_points(d, 'legendre'))

    # coefficient matrix for formulating the collocation equation
    C = np.zeros((d + 1, d + 1))

    # coefficient matrix for formulating the collocation equation
    D = np.zeros(d + 1)

    # coefficient matrix for formulating the collocation equation
    B = np.zeros(d + 1)

    # construct polynomial basis
    for j in range(d + 1):
        # construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

        # evaluate polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # evaluate time derivative of polynomial at collocation points to get the coefficients of continuity equation
        p_der = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = p_der(tau[r])

        # evaluate integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    # ------------------------------------------------------------------------------------------------------------------
    # STATE VARIABLES --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    nx = 5

    # velocity [m/s]
    v_n = ca.SX.sym('v_n')
    v_s = 50
    v = v_s * v_n

    # side slip angle [rad]
    beta_n = ca.SX.sym('beta_n')
    beta_s = 0.5
    beta = beta_s * beta_n

    # yaw rate [rad/s]
    omega_z_n = ca.SX.sym('omega_z_n')
    omega_z_s = 1
    omega_z = omega_z_s * omega_z_n

    # lateral distance to reference line (positive = left) [m]
    n_n = ca.SX.sym('n_n')
    n_s = 5.0
    n = n_s * n_n

    # relative angle to tangent on reference line [rad]
    xi_n = ca.SX.sym('xi_n')
    xi_s = 1.0
    xi = xi_s * xi_n

    # scaling factors for state variables
    x_s = np.array([v_s, beta_s, omega_z_s, n_s, xi_s])

    # put all states together
    x = ca.vertcat(v_n, beta_n, omega_z_n, n_n, xi_n)

    # ------------------------------------------------------------------------------------------------------------------
    # CONTROL VARIABLES ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # number of control variables
    nu = 4

    # steer angle [rad]
    delta_n = ca.SX.sym('delta_n')
    delta_s = 0.5
    delta = delta_s * delta_n

    # positive longitudinal force (drive) [N]
    f_drive_n = ca.SX.sym('f_drive_n')
    f_drive_s = 7500.0
    f_drive = f_drive_s * f_drive_n

    # negative longitudinal force (brake) [N]
    f_brake_n = ca.SX.sym('f_brake_n')
    f_brake_s = 20000.0
    f_brake = f_brake_s * f_brake_n

    # lateral wheel load transfer [N]
    gamma_y_n = ca.SX.sym('gamma_y_n')
    gamma_y_s = 5000.0
    gamma_y = gamma_y_s * gamma_y_n

    # scaling factors for control variables
    u_s = np.array([delta_s, f_drive_s, f_brake_s, gamma_y_s])

    # put all controls together
    u = ca.vertcat(delta_n, f_drive_n, f_brake_n, gamma_y_n)

    # ------------------------------------------------------------------------------------------------------------------
    # MODEL EQUATIONS --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # general constants
    g = 9.81
    mass = veh.M

    lf = veh.lf     # front wheelbase length (from COM to front axle)
    lr = veh.lr     # rear  wheelbase length
    L = lf + lr     # total wheelbase length
    wf = veh.wf     # front axle width (distance between centers of wheels)
    wr = veh.wr     # rear axle width


    c_roll = 0      # rolling resistance coefficient
    cgz = veh.h     # vehicle center of mass in the z
    k_roll = 0.5    # front-rear balance of lateral weight transfer

    # Aero stuff
    air_density = veh.rho
    frontal_area = veh.A
    Cl = -0.5       # NEGATIVE Cl means downforce
    Cd = -0.5       # negative Cd means drag

    drag_coeff = 0.5*air_density*Cl*frontal_area
    df_coeff = - 0.5*air_density*Cd*frontal_area    # "Positive force" = down
    Cp = 0.5    # front-rear balance of downforce

    # curvature of reference line [rad/m]
    kappa = ca.SX.sym('kappa')

    # drag force [N]
    f_xdrag = drag_coeff * v ** 2

    # rolling resistance forces [N]
    f_xroll_fl = 0.5 * c_roll * mass * g * lr / L
    f_xroll_fr = 0.5 * c_roll * mass * g * lr / L
    f_xroll_rl = 0.5 * c_roll * mass * g * lf / L
    f_xroll_rr = 0.5 * c_roll * mass * g * lf / L
    f_xroll = c_roll * mass * g

    # static normal tire forces [N]
    f_zstat_fl = 0.5 * mass * g * lr / L
    f_zstat_fr = 0.5 * mass * g * lr / L
    f_zstat_rl = 0.5 * mass * g * lf / L
    f_zstat_rr = 0.5 * mass * g * lf / L

    # dynamic normal tire forces (aerodynamic downforces) [N]
    f_zlift_fl = 0.5 * Cp*df_coeff * v ** 2
    f_zlift_fr = 0.5 * Cp*df_coeff * v ** 2
    f_zlift_rl = 0.5 * (1-Cp)*df_coeff * v ** 2
    f_zlift_rr = 0.5 * (1-Cp)*df_coeff * v ** 2

    # dynamic normal tire forces (load transfers) [N]
    f_zdyn_fl = (-0.5 * cgz / L * (f_drive + f_brake - f_xdrag - f_xroll)
                 - k_roll * gamma_y)
    f_zdyn_fr = (-0.5 * cgz / L * (f_drive + f_brake - f_xdrag - f_xroll)
                 + k_roll * gamma_y)
    f_zdyn_rl = (0.5 * cgz / L * (f_drive + f_brake - f_xdrag - f_xroll)
                 - (1.0 - k_roll) * gamma_y)
    f_zdyn_rr = (0.5 * cgz / L * (f_drive + f_brake - f_xdrag - f_xroll)
                 + (1.0 - k_roll) * gamma_y)

    # sum of all normal tire forces [N]
    f_z_fl = f_zstat_fl + f_zlift_fl + f_zdyn_fl
    f_z_fr = f_zstat_fr + f_zlift_fr + f_zdyn_fr
    f_z_rl = f_zstat_rl + f_zlift_rl + f_zdyn_rl
    f_z_rr = f_zstat_rr + f_zlift_rr + f_zdyn_rr

    # slip angles [rad]
    alpha_fl = delta - ca.atan((v * ca.sin(beta) + lf * omega_z) /
                               (v * ca.cos(beta) - 0.5 * wf * omega_z))
    alpha_fr = delta - ca.atan((v * ca.sin(beta) + lf * omega_z) /
                               (v * ca.cos(beta) + 0.5 * wf * omega_z))
    alpha_rl = ca.atan((-v * ca.sin(beta) + lr * omega_z) /
                       (v * ca.cos(beta) - 0.5 * wr * omega_z))
    alpha_rr = ca.atan((-v * ca.sin(beta) + lr * omega_z) /
                       (v * ca.cos(beta) + 0.5 * wr * omega_z))

    
    # lateral and longitudinal tire forces in the wheel frame
    f_x_fl, f_y_fl = combined_slip_forces(0, alpha_fl, f_z_fl)
    f_x_fr, f_y_fr = combined_slip_forces(0, alpha_fr, f_z_fr)  
    f_x_rl, f_y_rl = combined_slip_forces(0, alpha_rl, f_z_rl)
    f_x_rr, f_y_rr = combined_slip_forces(0, alpha_rr, f_z_rr)
   
    # longitudinal acceleration [m/s²]
    ax = (f_x_rl + f_x_rr + (f_x_fl + f_x_fr) * ca.cos(delta) - (f_y_fl + f_y_fr) * ca.sin(delta) - f_xdrag) / mass

    # lateral acceleration [m/s²]
    ay = ((f_x_fl + f_x_fr) * ca.sin(delta) + f_y_rl + f_y_rr + (f_y_fl + f_y_fr) * ca.cos(delta)) / mass


    # ------------------------------------------------------------------------------------------------------------------
    # DERIVATIVES ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # time-distance scaling factor (dt/ds)
    sf = (1.0 - n * kappa) / (v * (ca.cos(xi + beta)))

    # model equations for two track model (ordinary differential equations)
    dv = (sf / mass) * ((f_x_rl + f_x_rr) * ca.cos(beta) + (f_x_fl + f_x_fr) * ca.cos(delta - beta)
                        + (f_y_rl + f_y_rr) * ca.sin(beta) - (f_y_fl + f_y_fr) * ca.sin(delta - beta)
                        - f_xdrag * ca.cos(beta))

    dbeta = sf * (-omega_z + (-(f_x_rl + f_x_rr) * ca.sin(beta) + (f_x_fl + f_x_fr) * ca.sin(delta - beta)
                              + (f_y_rl + f_y_rr) * ca.cos(beta) + (f_y_fl + f_y_fr) * ca.cos(delta - beta)
                              + f_xdrag * ca.sin(beta)) / (mass * v))

    domega_z = (sf / veh.Iz) * ((f_x_rr - f_x_rl) * lr / 2
                                    - (f_y_rl + f_y_rr) * wr
                                    + ((f_x_fr - f_x_fl) * ca.cos(delta)
                                       + (f_y_fl - f_y_fr) * ca.sin(delta)) * wf / 2
                                    + ((f_y_fl + f_y_fr) * ca.cos(delta)
                                       + (f_x_fl + f_x_fr) * ca.sin(delta)) * wf)

    dn = sf * v * ca.sin(xi + beta)

    dxi = sf * omega_z - kappa

    # ODEs: driving dynamics only
    dx = ca.vertcat(dv, dbeta, domega_z, dn, dxi) / x_s

    # ------------------------------------------------------------------------------------------------------------------
    # CONTROL BOUNDARIES -----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    delta_min = -veh.delta_max / delta_s            # min. steer angle [rad]
    delta_max = veh.delta_max / delta_s             # max. steer angle [rad]
    f_drive_min = 0.0                               # min. longitudinal drive force [N]
    f_drive_max =  veh.drive_max / f_drive_s        # max. longitudinal drive force [N]
    f_brake_min = -veh.brake_max / f_brake_s        # min. longitudinal brake force [N]
    f_brake_max = 0.0                               # max. longitudinal brake force [N]
    gamma_y_min = -np.inf                           # min. lateral wheel load transfer [N]
    gamma_y_max = np.inf                            # max. lateral wheel load transfer [N]

    # ------------------------------------------------------------------------------------------------------------------
    # STATE BOUNDARIES -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    v_min = 1.0 / v_s                               # min. velocity [m/s]
    v_max = veh.max_velocity / v_s                  # max. velocity [m/s]
    beta_min = -0.5 * np.pi / beta_s                # min. side slip angle [rad]
    beta_max = 0.5 * np.pi / beta_s                 # max. side slip angle [rad]
    omega_z_min = - 0.5 * np.pi / omega_z_s         # min. yaw rate [rad/s]
    omega_z_max = 0.5 * np.pi / omega_z_s           # max. yaw rate [rad/s]
    xi_min = - 0.5 * np.pi / xi_s                   # min. relative angle to tangent on reference line [rad]
    xi_max = 0.5 * np.pi / xi_s                     # max. relative angle to tangent on reference line [rad]

    # an average f1 track is 12m wide -1.5m for the car, -1m buffer on each side
    n_min = -4 / n_s  # min lateral distance from reference line [m]
    n_max = 4 / n_s

    # ------------------------------------------------------------------------------------------------------------------
    # INITIAL GUESS FOR DECISION VARIABLES -----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    v_guess = 20.0 / v_s

    # ------------------------------------------------------------------------------------------------------------------
    # HELPER FUNCTIONS -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # continuous time dynamics
    f_dyn = ca.Function('f_dyn', [x, u, kappa], [dx, sf], ['x', 'u', 'kappa'], ['dx', 'sf'])

    # longitudinal tire forces [N]
    f_fx = ca.Function('f_fx', [x, u], [f_x_fl, f_x_fr, f_x_rl, f_x_rr],
                       ['x', 'u'], ['f_x_fl', 'f_x_fr', 'f_x_rl', 'f_x_rr'])
    # lateral tire forces [N]
    f_fy = ca.Function('f_fy', [x, u], [f_y_fl, f_y_fr, f_y_rl, f_y_rr],
                       ['x', 'u'], ['f_y_fl', 'f_y_fr', 'f_y_rl', 'f_y_rr'])
    # vertical tire forces [N]
    f_fz = ca.Function('f_fz', [x, u], [f_z_fl, f_z_fr, f_z_rl, f_z_rr],
                       ['x', 'u'], ['f_z_fl', 'f_z_fr', 'f_z_rl', 'f_z_rr'])

    # longitudinal and lateral acceleration [m/s²]
    f_a = ca.Function('f_a', [x, u], [ax, ay], ['x', 'u'], ['ax', 'ay'])

    # ------------------------------------------------------------------------------------------------------------------
    # FORMULATE NONLINEAR PROGRAM --------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # initialize NLP vectors
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    cat = []
    g = []
    lbg = []
    ubg = []

    # initialize ouput vectors
    x_opt = []
    u_opt = []
    dt_opt = []
    tf_opt = []
    ax_opt = []
    ay_opt = []
    ec_opt = []
    k_opt = []

    # initialize control vectors (for regularization)
    delta_p = []
    F_p = []

    # boundary constraint: lift initial conditions
    Xk = ca.MX.sym('X0', nx)
    w.append(Xk)
    lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min])
    ubw.append([v_max, beta_max, omega_z_max, n_max, xi_max])
    w0.append([v_guess, 0.0, 0.0, 0.0, 0.0])
    x_opt.append(Xk * x_s)

    # loop along the racetrack and formulate path constraints & system dynamic
    # retrieve step-sizes of optimization along reference line
    h = mesh_point_separations
    for k in range(N):
        # add decision variables for the control
        Uk = ca.MX.sym('U_' + str(k), nu)
        w.append(Uk)
        lbw.append([delta_min, f_drive_min, f_brake_min, gamma_y_min])
        ubw.append([delta_max, f_drive_max, f_brake_max, gamma_y_max])
        w0.append([0.0] * nu)

        # add decision variables for the state at collocation points
        Xc = []
        for j in range(d):
            Xkj = ca.MX.sym('X_' + str(k) + '_' + str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-np.inf] * nx)
            ubw.append([np.inf] * nx)
            w0.append([v_guess, 0.0, 0.0, 0.0, 0.0])

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

            # append collocation equations (system dynamic); added dynamic tolerance for better convergence
            dynamic_tol = 1e-6
            fj, qj = f_dyn(Xc[j - 1], Uk, kappa_col)
            cat.extend(['System dynamic']*nx)
            g.append(h[k] * fj - xp)
            lbg.append([-dynamic_tol] * nx)
            ubg.append([dynamic_tol] * nx)

            # add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j - 1]

            # add contribution to quadrature function
            J = J + B[j] * qj * h[k]

            # add contribution to scaling factor (for calculating lap time)
            sf_opt.append(B[j] * qj * h[k])
            curv.append(kappa_col)

        
        # calculate used energy 
        k_opt.append(curv[1])
        dt_opt.append(sf_opt[0] + sf_opt[1] + sf_opt[2])
        ec_opt.append(Xk[0] * v_s * Uk[1] * f_drive_s * dt_opt[-1])

        # add new decision variables for state at end of the collocation interval
        Xk = ca.MX.sym('X_' + str(k + 1), nx)
        w.append(Xk)
        lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min])
        ubw.append([v_max, beta_max, omega_z_max, n_max, xi_max])
        w0.append([v_guess, 0.0, 0.0, 0.0, 0.0])

        # add equality constraint
        cat.extend(['Equality constraint']*nx)
        g.append(Xk_end - Xk)
        lbg.append([0.0] * nx)
        ubg.append([0.0] * nx)

        # get tire forces
        f_x_flk, f_x_frk, f_x_rlk, f_x_rrk = f_fx(Xk, Uk)
        f_y_flk, f_y_frk, f_y_rlk, f_y_rrk = f_fy(Xk, Uk)
        f_z_flk, f_z_frk, f_z_rlk, f_z_rrk = f_fz(Xk, Uk)

        # get accelerations (longitudinal + lateral)
        axk, ayk = f_a(Xk, Uk)

        # path constraint: limitied engine power
        powercap = 60000 # 60 kW
        cat.extend(['Limited power constraint'])
        g.append(Xk[0] * Uk[1])
        lbg.append([-np.inf])
        ubg.append([powercap / (f_drive_s * v_s)])

        # get constant friction coefficient
        mue_fl = mu
        mue_fr = mu
        mue_rl = mu
        mue_rr = mu

        # path constraint: Kamm's Circle for each wheel
        cat.extend(['Kamm Circle']*4)
        g.append(((f_x_flk / (mue_fl * f_z_flk)) ** 2 + (f_y_flk / (mue_fl * f_z_flk)) ** 2))
        g.append(((f_x_frk / (mue_fr * f_z_frk)) ** 2 + (f_y_frk / (mue_fr * f_z_frk)) ** 2))
        g.append(((f_x_rlk / (mue_rl * f_z_rlk)) ** 2 + (f_y_rlk / (mue_rl * f_z_rlk)) ** 2))
        g.append(((f_x_rrk / (mue_rr * f_z_rrk)) ** 2 + (f_y_rrk / (mue_rr * f_z_rrk)) ** 2))
        lbg.append([0.0] * 4)
        ubg.append([1.0] * 4)

        # path constraint: lateral wheel load transfer
        cat.extend(['Lateral load transfer'])
        g.append(((f_y_flk + f_y_frk) * ca.cos(Uk[0] * delta_s) + f_y_rlk + f_y_rrk
                  + (f_x_flk + f_x_frk) * ca.sin(Uk[0] * delta_s))
                 * cgz / ((wf + wr) / 2) - Uk[3] * gamma_y_s)
        lbg.append([0.0])
        ubg.append([0.0])

        # path constraint: f_drive * f_brake == 0 (no simultaneous operation of brake and accelerator pedal)
        cat.extend(['Simultaneous drive and brake'])
        g.append(Uk[1] * Uk[2])
        lbg.append([-20000.0 / (f_drive_s * f_brake_s)])
        ubg.append([0.0])

        # # path constraint: actor dynamic
        # if k > 0:
        #     sigma = (1 - kappa_interp(k) * Xk[3] * n_s) / (Xk[0] * v_s)
        #     g.append((Uk - w[1 + (k - 1) * (nx)]) / (h[k - 1] * sigma))
        #     lbg.append([delta_min / (veh["t_delta"]), -np.inf, f_brake_min / (veh["t_brake"]), -np.inf])
        #     ubg.append([delta_max / (veh["t_delta"]), f_drive_max / (veh["t_drive"]), np.inf, np.inf])

        # # path constraint: safe trajectories with acceleration ellipse
        # if pars["optim_opts"]["safe_traj"]:
        #     g.append((ca.fmax(axk, 0) / pars["optim_opts"]["ax_pos_safe"]) ** 2
        #              + (ayk / pars["optim_opts"]["ay_safe"]) ** 2)
        #     g.append((ca.fmin(axk, 0) / pars["optim_opts"]["ax_neg_safe"]) ** 2
        #              + (ayk / pars["optim_opts"]["ay_safe"]) ** 2)
        #     lbg.append([0.0] * 2)
        #     ubg.append([1.0] * 2)

        # append controls (for regularization)
        delta_p.append(Uk[0] * delta_s)
        F_p.append(Uk[1] * f_drive_s / 10000.0 + Uk[2] * f_brake_s / 10000.0)

        # append outputs
        x_opt.append(Xk * x_s)
        u_opt.append(Uk * u_s)
        tf_opt.extend([f_x_flk, f_y_flk, f_z_flk, f_x_frk, f_y_frk, f_z_frk])
        tf_opt.extend([f_x_rlk, f_y_rlk, f_z_rlk, f_x_rrk, f_y_rrk, f_z_rrk])
        ax_opt.append(axk)
        ay_opt.append(ayk)


    # boundary constraint: start states = final states
    cat.extend(['Initial = final']*nx)
    g.append(w[0] - Xk)
    lbg.append([0.0, 0.0, 0.0, 0.0, 0.0])
    ubg.append([0.0, 0.0, 0.0, 0.0, 0.0])

    # # path constraint: limited energy consumption
    # if pars["optim_opts"]["limit_energy"]:
    #     g.append(ca.sum1(ca.vertcat(*ec_opt)) / 3600000.0)
    #     lbg.append([0])
    #     ubg.append([pars["optim_opts"]["energy_limit"]])

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

    # formulate objective

    r_delta = 10 
    r_F = 0.5 *10**-8
    J = J + r_F * Jp_f + r_delta * Jp_delta

    # concatenate NLP vectors
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    cat = np.array(cat)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    # concatenate output vectors
    x_opt = ca.vertcat(*x_opt)
    u_opt = ca.vertcat(*u_opt)
    tf_opt = ca.vertcat(*tf_opt)
    dt_opt = ca.vertcat(*dt_opt)
    ax_opt = ca.vertcat(*ax_opt)
    ay_opt = ca.vertcat(*ay_opt)
    ec_opt = ca.vertcat(*ec_opt)

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE NLP SOLVER ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # fill nlp structure
    nlp = {'f': J, 'x': w, 'g': g}

    # solver options
    opts = {"expand": True,
            "ipopt.max_iter": 2000,
            "ipopt.tol": 1e-7,
            "ipopt.print_level": 5}


    # create solver instance
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)


    
    # Evaluate constraint violations at the initial guess
    violation_threshold = 3e-4      # ignore constraint violations smaller than this to filter noise
    g_eval = ca.Function('g_eval', [w], [g])  # Create function to evaluate constraints

    g_violations = np.array(g_eval(w0))  # Compute constraint violations
    violated_constraints = np.where((g_violations < lbg - violation_threshold) | (g_violations > ubg + violation_threshold))[0]

    print(f"Total constraints: {len(g_violations)}")
    print(f"Number of violated constraints at initial guess: {len(violated_constraints)}")

    buffer = 0
    if len(violated_constraints) > 0:

        max_violation = np.max(np.abs(g_violations[violated_constraints]))

        print("Violated constraints indices and values:")
        for idx in violated_constraints:
            buffer += 1
            if buffer % 1000 == 0:
                print(f"Constraint {idx} / {cat[idx]}: {g_violations[idx]}, Expected: [{lbg[idx]}, {ubg[idx]}]")


        print(f"Max violation at initial guess: {max_violation:.6f}")

    # ------------------------------------------------------------------------------------------------------------------
    # SOLVE NLP --------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # start time measure
    t0 = time.perf_counter()

    # solve NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    # end time measure
    tend = time.perf_counter()

    if solver.stats()['return_status'] != 'Solve_Succeeded':
        print('\033[91m' + 'ERROR: Optimization did not succeed!' + '\033[0m')
        sys.exit(1)

    # ------------------------------------------------------------------------------------------------------------------
    # EXTRACT SOLUTION -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # helper function to extract solution for state variables, control variables, tire forces, time
    f_sol = ca.Function('f_sol', [w], [x_opt, u_opt, tf_opt, dt_opt, ax_opt, ay_opt, ec_opt],
                        ['w'], ['x_opt', 'u_opt', 'tf_opt', 'dt_opt', 'ax_opt', 'ay_opt', 'ec_opt'])

    # extract solution
    x_opt, u_opt, tf_opt, dt_opt, ax_opt, ay_opt, ec_opt = f_sol(sol['x'])

    # solution for state variables
    x_opt = np.reshape(x_opt, (N + 1, nx))

    # solution for control variables
    u_opt = np.reshape(u_opt, (N, nu))

    # solution for tire forces
    tf_opt = np.append(tf_opt[-12:], tf_opt[:])
    tf_opt = np.reshape(tf_opt, (N + 1, 12))

    # solution for time
    t_opt = np.hstack((0.0, np.cumsum(dt_opt)))

    print("Optimal laptime is: ", t_opt)

    # solution for acceleration
    ax_opt = np.append(ax_opt[-1], ax_opt)
    ay_opt = np.append(ay_opt[-1], ay_opt)
    atot_opt = np.sqrt(np.power(ax_opt, 2) + np.power(ay_opt, 2))

    # solution for energy consumption
    ec_opt_cum = np.hstack((0.0, np.cumsum(ec_opt))) / 3600.0

    # solution for curvature
    k_opt = np.array(k_opt)
    k_opt = np.append(k_opt[-1], k_opt)
    
    df0 = np.column_stack(
        (t_opt, k_opt, x_opt, u_opt)
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
    ]

    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter("bicycle_out.xlsx", engine="xlsxwriter")

    # Write the dataframe to a excel sheet
    df1.to_excel(writer, sheet_name="Sheet1", index=False, header=header)

    # Close the Pandas Excel writer and output the excel file
    writer.close()

    print("Optimization took: {:.2f} seconds".format(tend-t0))

opt_mintime()