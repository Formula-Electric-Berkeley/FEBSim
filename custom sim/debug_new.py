import os
import sys
import time
import numpy as np
import casadi as ca
import pandas as pd

import track as tr
import test_track as tst
import vehicle as veh

#helper functions for casasdi models
def smooth_abs(val, eps = 1e-9):
        return ca.sqrt(val**2 + eps**2)

def smooth_sign(val, eps = 1e-9):
        return val / smooth_abs(val)

#Define constants
mu      = 0.75        # coefficient of friction for tires
# pacejka model
Bx = 16
Cx = 1.58
Ex = 0.1

By = 13
Cy = 1.45
Ey = -0.8
#TODO: replace with Nick's tire model later
#This model can handle slip ratios outside [-1, 1], but we shouldn't get there
def combined_slip_forces(s,a,Fn):        
        Fx0 = mu*Fn*ca.sin(Cx*ca.arctan((1-Ex)*Bx*s + Ex*ca.arctan(Bx * s)))
        Fy0 = mu*Fn*ca.sin(Cy*ca.arctan((1-Ey)*By*a + Ey*ca.arctan(By * a)))

        Exa =-Fn/10000
        Dxa = 1
        Cxa = 1
        Bxa = 13 * ca.cos(ca.arctan(9.7*s))
        Gxa = Dxa*ca.cos(Cxa*ca.arctan(Bxa*a - Exa*(Bxa*a - ca.arctan(Bxa*a))))

        Eys = 0.3
        Dys = 1
        Cys = 1
        Bys = 10.62*ca.cos(ca.arctan(7.8*a))
        Gys = Dys*ca.cos(Cys*ca.arctan(Bys*s - Eys*(Bys*s - ca.arctan(Bys*s))))

        Fx  = Fx0 * Gxa
        Fy  = Fy0 * Gys
        return Fx,Fy


def opt_mintime():
        # ------------------------------------------------------------------------------------------------------------------
        # GET TRACK INFORMATION --------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------


        
        #kappa_refline_cl = np.abs(tr.segments[:, 1])
        kappa_refline_cl = tr.curvatures

        discr_points = len(tr.curvatures)#segments)
        w_tr_left_cl = tr.track_widths
        w_tr_right_cl = tr.track_widths



        # optimization steps (0, 1, 2 ... end point/start point)
        steps = [i for i in range(discr_points)]
        N = steps[-1] 

        # step size and separation along the reference line
        h = tr.mesh_size    
        s_opt = np.ones(discr_points) * h



        # interpolate track data from reference line to the number of steps sampled 
        kappa_interp = ca.interpolant('kappa_interp', 'linear', [steps], kappa_refline_cl) #curvature of reference line
        #w_tr_left_interp = ca.interpolant('w_tr_left_interp', 'linear', [steps], w_tr_left_cl) #distance from refline to track edge (left)
        #w_tr_right_interp = ca.interpolant('w_tr_right_interp', 'linear', [steps], w_tr_right_cl)

        # ------------------------------------------------------------------------------------------------------------------
        # DIRECT GAUSS-LEGENDRE COLLOCATION --------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        #This is taken directly from CasADi but seems fairly standard; the process is fully described in the documentation

        # degree of interpolating polynomial ("sweet spot" according to MIT robotics paper)
        d = 3

        # legendre collocation points
        tau = np.append(0, ca.collocation_points(d, 'legendre')) #directly from CasADi documentation

        # coefficient matrix for the collocation equation
        C = np.zeros((d + 1, d + 1))

        # coefficient matrix for the continuity equation
        D = np.zeros(d + 1)

        # coefficient matrix for the quadrature function
        B = np.zeros(d + 1)

        # construct polynomial basis
        for j in range(d + 1):
                # construct Lagrange polynomials to get the polynomial basis at the collocation point
                p = np.poly1d([1])
                for r in range(d + 1):
                        if r != j:
                                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r]) #exactly our p_kis as described in documentation (and on Wikipedia!)

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

        # number of state variables
        nx = 9 #5 from them, 4 wheel speed

        
        # longitudinal velocity [m/s]
        v_n = ca.SX.sym('v_n')
        v_s = 50
        v = v_s * v_n

        # lateral velocity [m/s]
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

        # wheel angular velocities
        wfr_n = ca.SX.sym('wfr_n')
        wfl_n = ca.SX.sym('wfl_n')
        wrl_n = ca.SX.sym('wrl_n')
        wrr_n = ca.SX.sym('wrr_n')

        wheel_scale = 300 #crucial for convergence
        wfr_s = wheel_scale
        wfl_s = wheel_scale
        wrr_s = wheel_scale
        wrl_s = wheel_scale

        wfr = wfr_s * wfr_n 
        wfl = wfl_s * wfl_n 
        wrr = wrr_s * wrr_n 
        wrl = wrl_s * wrl_n 

        x_s = np.array([v_s, beta_s, omega_z_s, n_s, xi_s, wfr_s, wfl_s, wrl_s, wrr_s])
        x = ca.vertcat(v_n, beta_n, omega_z_n, n_n, xi_n, wfr_n, wfl_n, wrl_n, wrr_n)

        # ------------------------------------------------------------------------------------------------------------------
        # CONTROL VARIABLES ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # number of control variables
        nu = 5

        # steer angle [rad]
        delta_n = ca.SX.sym('delta_n')
        delta_s = 0.5
        delta = delta_s * delta_n

        # positive longitudinal force (drive) [N-m]
        torque_drive_n = ca.SX.sym('f_drive_n')
        torque_drive_s = 6000.0
        torque_drive = torque_drive_s * torque_drive_n

        # negative longitudinal force (brake) [N]
        f_brake_n = ca.SX.sym('f_brake_n')
        f_brake_s = 20000.0
        f_brake = f_brake_s * f_brake_n

        # longitudinal wheel load transfer [N]
        gamma_x_n = ca.SX.sym('gamma_x_n')
        gamma_x_s = 5000.0
        gamma_x = gamma_x_s * gamma_x_n

        # lateral wheel load transfer [N]
        gamma_y_n = ca.SX.sym('gamma_y_n')
        gamma_y_s = 5000.0
        gamma_y = gamma_y_s * gamma_y_n

        # curvature of reference line [rad/m]
        kappa = ca.SX.sym('kappa')      #no real good spot to define this; not a state or a control really

        # scaling factors for control variables
        #u_s = np.array([delta_s, torque_drive_s, f_brake_s])

        # put all controls together
        #u = ca.vertcat(delta_n, torque_drive_n, f_brake_n)

        # scaling factors for control variables
        u_s = np.array([delta_s, torque_drive_s, f_brake_s, gamma_x_s, gamma_y_s])

        # put all controls together
        u = ca.vertcat(delta_n, torque_drive_n, f_brake_n, gamma_x_n, gamma_y_n)

        # ------------------------------------------------------------------------------------------------------------------
        # MODEL PHYSICS ----------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # general constants
        g       = 9.81           #[m/s^2]
        mass    = veh.M          #[kg]
        lf      = veh.lf         # front wheelbase length
        lr      = veh.lr         # rear  wheelbase length
        L       = lf + lr        # total wheelbase length
        wf      = veh.wf         # rear axle width
        wr      = veh.wf         # front axle width


        Je      = veh.Je        # engine moment of inertia
        Jw      = veh.Jw        # wheel moment of inertia
        re      = veh.re        # effective wheel radius
        rb      = veh.rb         # effective brake radius
        Iz      = veh.Iz        # vehicle moment of inertia about z axis


        #TODO@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        '''
        Try copying their code and using a different longitudinal force model
        
        '''



        # reframe state to be convenient for UMich formulations (normal and tangent to the reference line)

        # side slip angle, angle from longitudinal vehicle axis to velocity [rad]
        vx = v*ca.cos(beta)
        vy = v*ca.sin(beta)
        

        # compute normal forces, assumes wf ~ wr
        FNfr = mass*g*lr/L/2 + gamma_x - gamma_y
        FNfl = mass*g*lr/L/2 + gamma_x + gamma_y
        FNrr = mass*g*lf/L/2 - gamma_x - gamma_y
        FNrl = mass*g*lf/L/2 - gamma_x + gamma_y


        # compute body frame components of linear velocity at each wheel
        vxfr = vx + wf/2*omega_z
        vyfr = vy + lf  *omega_z
        vxfl = vx - wf/2*omega_z
        vyfl = vy + lf  *omega_z
        vxrr = vx + wr/2*omega_z
        vyrr = vy - lr  *omega_z
        vxrl = vx - wr/2*omega_z
        vyrl = vy - lr  *omega_z

        # compute wheel frame velocity components
        vlfr = vxfr * ca.cos(delta) + vyfr * ca.sin(delta)
        vtfr = vyfr * ca.cos(delta) - vxfr * ca.sin(delta)
        vlfl = vxfl * ca.cos(delta) + vyfl * ca.sin(delta)
        vtfl = vyfl * ca.cos(delta) - vxfl * ca.sin(delta)
        vlrr = vxrr #no rear steering
        vtrr = vyrr 
        vlrl = vxrl 
        vtrl = vyrl 

        # compute slip angles
        alpha_fr = -ca.arctan(vtfr / smooth_abs(vlfr))
        alpha_fl = -ca.arctan(vtfl / smooth_abs(vlfl))
        alpha_rr = -ca.arctan(vtrr / smooth_abs(vlrr))
        alpha_rl = -ca.arctan(vtrl / smooth_abs(vlrl))

        # compute slip ratios
        sigma_fr = (re * wfr - vlfr) / smooth_abs(vlfr)
        sigma_fl = (re * wfl - vlfl) / smooth_abs(vlfl)
        sigma_rr = (re * wrr - vlrr) / smooth_abs(vlrr)
        sigma_rl = (re * wrl - vlrl) / smooth_abs(vlrl)
        #Works without slip ratios!

        # compute wheel traction forces in wheel frame
        Flfr, Ftfr = combined_slip_forces(sigma_fr, alpha_fr, FNfr)
        Flfl, Ftfl = combined_slip_forces(sigma_fl, alpha_fl, FNfl)
        Flrr, Ftrr = combined_slip_forces(sigma_rr, alpha_rr, FNrr)
        Flrl, Ftrl = combined_slip_forces(sigma_rl, alpha_rl, FNrl)

        # change wheel forces to body frame
        Fxfr = Flfr * ca.cos(delta) - Ftfr * ca.sin(delta)
        Fyfr = Flfr * ca.sin(delta) + Ftfr * ca.cos(delta)
        Fxfl = Flfl * ca.cos(delta) - Ftfl * ca.sin(delta)
        Fyfl = Flfl * ca.sin(delta) + Ftfl * ca.cos(delta)
        Fxrr = Flrr 
        Fyrr = Ftrr 
        Fxrl = Flrl 
        Fyrl = Ftrl 

        # drag forces
        Fareox = - veh.drag_coeff * vx * smooth_abs(vx)

        # net force and torque on the vehicle
        Fx = Fxfr + Fxfl + Fxrl + Fxrr + Fareox #net force in the x
        Fy = Fyfr + Fyfl + Fyrr + Fyrl          #net force in the y
        Kz = lf*(Fyfr + Fyfl) - lr*(Fyrr + Fyrl) + wf/2*(Fxfr - Fxfl) + wr/2*(Fxrr - Fxrl)      #net torque about z axis


        ### PART B: WHEEL EQUATIONS OF MOTION
        Fbrake_fr = veh.brake_fr*f_brake
        Fbrake_fl = veh.brake_fl*f_brake
        Fbrake_rl = veh.brake_rl*f_brake
        Fbrake_rr = veh.brake_rr*f_brake

        # wheel dynamics
        Tbrake_fr = rb * Fbrake_fr #* smooth_sign(wfr)
        Tbrake_fl = rb * Fbrake_fl #* smooth_sign(wfl)
        Tbrake_rr = rb * Fbrake_rr #* smooth_sign(wrr)
        Tbrake_rl = rb * Fbrake_rl #* smooth_sign(wrl)
        
        #TODO: why would the wheels ever spin backwards? If they slip while we brake?
        

        # ------------------------------------------------------------------------------------------------------------------
        # DERIVATIVES ------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------


        # time-distance scaling factor (dt/ds) -> this is what lets us not integrate in t!
        sf = (1.0 - n * kappa) / (v * (ca.cos(xi + beta)))
        #assume this is non-negative for now; we must define n and kappa more rigorously later

        vx_dot =  (vy*omega_z + Fx/mass)*sf #includes Newton-Euler pseudoforces due to rotation
        vy_dot = (-vx*omega_z + Fy/mass)*sf

        dv = (sf / mass) * ((Flrl + Flrr) * ca.cos(beta) + (Flfl + Flfr) * ca.cos(delta - beta)
                        + (Ftrl + Ftrr) * ca.sin(beta) - (Ftfl + Ftfr) * ca.sin(delta - beta)
                        - Fareox * ca.cos(beta))

        dbeta = sf * (-omega_z + (-(Flrl + Flrr) * ca.sin(beta) + (Flfl + Flfr) * ca.sin(delta - beta)
                              + (Ftrl + Ftrr) * ca.cos(beta) + (Ftfl + Ftfr) * ca.cos(delta - beta)
                              + Fareox * ca.sin(beta)) / (mass * v))


        omegaz_dot = sf*Kz/Iz

        xi_dot = sf*omega_z - kappa
        n_dot = sf * v * ca.sin(xi + beta)

        #Convert body-frame dynamics to earth-frame
        #psi_dot = omega_z #psi = angle between vehicle x and Earth X axes
        #Initially, psi = 0; axes are aligned. Cos(0) = 1
        #X_dot = v*ca.cos(psi)
        #Y_dot = v*ca.sin(psi)

        #Wheel speed dynamics
        #TODO: validate/alter the formatting of this: Jw and torque_brake?
        wfr_dot = sf*(-Flfr*re + Tbrake_fr) / Jw
        wfl_dot = sf*(-Flfl*re + Tbrake_fl) / Jw
        wrr_dot = sf*(-Flrr*re + Tbrake_rr + torque_drive/2) / (Jw + Je*veh.ratio_final/2) #gear ratio
        wrl_dot = sf*(-Flrl*re + Tbrake_rl + torque_drive/2) / (Jw + Je*veh.ratio_final/2)


        # ODEs: driving dynamics 
        dx = ca.vertcat(dv, dbeta, omegaz_dot, n_dot, xi_dot, wfr_dot, wfl_dot, wrl_dot, wrr_dot) / x_s

        # ------------------------------------------------------------------------------------------------------------------
        # CONTROL BOUNDARIES -----------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        #Recall:         u = (delta, torque_drive, f_brake)

        delta_min = -veh.delta_max / delta_s            # min. steer angle [rad]
        delta_max = veh.delta_max / delta_s             # max. steer angle [rad]
        f_drive_min = 0.0                               # min. longitudinal drive TORQUE [Nm]
        f_drive_max = veh.drive_max / torque_drive_s    # max. longitudinal drive torque [Nm]
        f_brake_min = -veh.brake_max / f_brake_s        # min. longitudinal brake force [N]
        f_brake_max = 0.0                               # max. longitudinal brake force [N]
        gamma_x_min = -np.inf                           # min. longitudinal wheel load transfer [N]
        gamma_x_max = np.inf                            # max. longitudinal wheel load transfer [N]

        gamma_y_min = -np.inf                           # min. lateral wheel load transfer [N]
        gamma_y_max = np.inf                            # max. lateral wheel load transfer [N]

        #TODO
        #test if we're violating these indivual constraints
        #Set everything to infinity
        #Check sign of brake force

        # ------------------------------------------------------------------------------------------------------------------
        # STATE BOUNDARIES -------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        #State format:   x = (vx, vy, omega_z, n, xi, wfr, wfl, wrl, wrr)
        v_min = 0.0 / v_s                               # min. velocity [m/s]
        v_max = veh.max_velocity / v_s                  # max. velocity [m/s]
        beta_min = -0.5 * np.pi / beta_s                # min. side slip angle [rad]
        beta_max = 0.5 * np.pi / beta_s                 # max. side slip angle [rad]
        omega_z_min = - 0.5 * np.pi / omega_z_s         # min. yaw rate [rad/s]
        omega_z_max = 0.5 * np.pi / omega_z_s           # max. yaw rate [rad/s]
        xi_min = - 0.5 * np.pi / xi_s                   # min. relative angle to tangent on reference line [rad]
        xi_max = 0.5 * np.pi / xi_s                     # max. relative angle to tangent on reference line [rad]

        #TODO check these also (v_max etc.)

        #TODO: this and the scalar for the wheel speed matter a lot
        wheel_bound = 200 #2*veh.max_velocity/veh.re                             # max. wheel speed [rad/s]; fixes slip angle to +/- 1
        #Note -- optimization DOES NOT SUCCEED if max is 1.5*v_max*veh.re; we need to decrease our scalar?
        wheelspeed_min = 0.0 
        wheelspeed_max = wheel_bound / wheel_scale                  

        # ------------------------------------------------------------------------------------------------------------------
        # INITIAL GUESS FOR DECISION VARIABLES -----------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        vx_guess = 20.0 / v_s

        # ------------------------------------------------------------------------------------------------------------------
        # HELPER FUNCTIONS -------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        #TODO: Look at these!***************************
        # continuous time dynamics
        f_dyn = ca.Function('f_dyn', [x, u, kappa], [dx, sf], ['x', 'u', 'kappa'], ['dx', 'sf'])

        # longitudinal tire forces in the vehicle frame [N]
        f_fx = ca.Function('f_fx', [x, u], [Fxfl, Fxfr, Fxrl, Fxrr],
                        ['x', 'u'], ['Fxfl', 'Fxfr', 'Fxrl', 'Fxrr'])
        # lateral tire forces in the vehicle frame [N]
        f_fy = ca.Function('f_fy', [x, u], [Fyfl, Fyfr, Fyrl, Fyrr],
                        ['x', 'u'], ['Fyfl', 'Fyfr', 'Fyrl', 'Fyrr'])


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
        #ax_opt = []
        #ay_opt = []
        #ec_opt = []

        # initialize control vectors (for regularization)
        delta_p = []
        F_p = []

        # boundary constraint: lift initial conditions
        Xk = ca.MX.sym('X0', nx)
        w.append(Xk)
        n_min = 0 #(-tr.track_widths[0] + veh.car_width / 2) / n_s
        n_max = 0 #(tr.track_widths[0] - veh.car_width / 2) / n_s
        
        lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min, wheelspeed_min, wheelspeed_min, wheelspeed_min, wheelspeed_min])
        ubw.append([v_max, beta_max, omega_z_max, n_max, xi_max, wheelspeed_max, wheelspeed_max, wheelspeed_max, wheelspeed_max])
        w0.append([vx_guess, 0.0, 0.0, 0.0, 0.0, vx_guess/veh.re, vx_guess/veh.re, vx_guess/veh.re, vx_guess/veh.re])
        x_opt.append(Xk * x_s)

        # loop along the racetrack and formulate path constraints & system dynamics
        # retrieve step-sizes of optimization along reference line
        h = s_opt
        for k in range(N):
                # add decision variables for the control
                Uk = ca.MX.sym('U_' + str(k), nu)
                w.append(Uk)
                lbw.append([delta_min, f_drive_min, f_brake_min, gamma_x_min, gamma_y_min])
                ubw.append([delta_max, f_drive_max, f_brake_max, gamma_x_max, gamma_y_max])
                w0.append([0.0]*nu)

                # add decision variables for the state at collocation points
                Xc = []
                for j in range(d):
                        Xkj = ca.MX.sym('X_' + str(k) + '_' + str(j), nx)
                        Xc.append(Xkj)
                        w.append(Xkj)
                        lbw.append([-np.inf] * nx)
                        ubw.append([np.inf] * nx)
                        w0.append([vx_guess, 0.0, 0.0, 0.0, 0.0, vx_guess/veh.re, vx_guess/veh.re, vx_guess/veh.re, vx_guess/veh.re])
                        #might need to reset initial guess to 0 for spin speeds bc they may be negative

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
                        g.append(h[k] * fj - xp)
                        lbg.append([0.0] * nx)
                        ubg.append([0.0] * nx)

                        # add contribution to the end state
                        Xk_end = Xk_end + D[j] * Xc[j - 1]

                        # add contribution to quadrature function
                        J = J + B[j] * qj * h[k]

                        # add contribution to scaling factor (for calculating lap time)
                        sf_opt.append(B[j] * qj * h[k])

                #calculate time step
                dt_opt.append(sf_opt[0] + sf_opt[1] + sf_opt[2])

                # add new decision variables for state at end of the collocation interval
                Xk = ca.MX.sym('X_' + str(k + 1), nx)
                w.append(Xk)
                n_min = 0 #(-tr.track_widths[k + 1] + veh.car_width / 2.0) / n_s
                n_max = 0 #(tr.track_widths[k + 1] - veh.car_width / 2.0) / n_s
                lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min, wheelspeed_min, wheelspeed_min, wheelspeed_min, wheelspeed_min])
                ubw.append([v_max, beta_max, omega_z_max, n_max, xi_max, wheelspeed_max, wheelspeed_max, wheelspeed_max, wheelspeed_max])
                w0.append([vx_guess, 0.0, 0.0, 0.0, 0.0, vx_guess/veh.re, vx_guess/veh.re, vx_guess/veh.re, vx_guess/veh.re])

                # add equality constraint; our chosen state at the next collocation point must = our predicted state when extrapolating
                g.append(Xk_end - Xk)
                lbg.append([0.0] * nx)
                ubg.append([0.0] * nx)

                # path constraint: f_drive * f_brake == 0 (no simultaneous operation of brake and accelerator pedal)
                g.append(Uk[1] * Uk[2])
                lbg.append([0.0]) #-20000.0 / (torque_drive_s * f_brake_s)])
                ubg.append([0.0])



                #TODO: when it converges to an infeasible solution, check what constraints it's violating
                

                # get tire forces
                f_x_flk, f_x_frk, f_x_rlk, f_x_rrk = f_fx(Xk, Uk)
                f_y_flk, f_y_frk, f_y_rlk, f_y_rrk = f_fy(Xk, Uk)

                f_xk = f_x_flk + f_x_frk + f_x_rlk + f_x_rrk
                f_yk = f_y_flk + f_y_frk + f_y_rlk + f_y_rrk

                # path constraint: longitudinal wheel load transfer
                g.append( Uk[3] * gamma_x_s *(veh.lf+veh.lr) + veh.cg_height*f_xk)
                #TODO: I wrote Fy here initially. This was wrong. Double check all physics
                lbg.append([0.0])
                ubg.append([0.0])

                # path constraint: lateral wheel load transfer
                g.append( Uk[4] * gamma_y_s *(veh.wf+veh.wr) + veh.cg_height*f_yk)
                lbg.append([0.0])
                ubg.append([0.0])

                # path constraint: actor dynamic (limit rate of change of steering and drive commands)
                delta_step = 4
                drive_step = 1 
                brake_step = 4
                if k > 0: #was k=0
                        sigma = (1 - kappa_interp(k) * Xk[3] * n_s) / (Xk[0] * v_s) #TODO: not exactly SF_k
                        g.append((Uk - w[1+(d+2)*(k-1)]) / (h[k - 1] * sigma))
                        lbg.append([delta_min / delta_step, -np.inf, f_brake_min / brake_step, -np.inf, -np.inf])
                        ubg.append([delta_max / delta_step, f_drive_max / drive_step, np.inf, np.inf, np.inf])

                
                # append controls (for regularization)
                delta_p.append(Uk[0] * delta_s)
                F_p.append(Uk[1] * torque_drive_s / 10000.0 + Uk[2] * f_brake_s / 10000.0)

                # append outputs
                x_opt.append(Xk * x_s)
                u_opt.append(Uk * u_s)

        
        # boundary constraint: start states = final states (closed track)
        #g.append(w[0] - Xk)
        #lbg.append([0.0, 0.0, 0.0, 0.0, 0.0])
        #ubg.append([0.0, 0.0, 0.0, 0.0, 0.0])


        #Regularization ---> ensures smoothing of control variables so we don't oscillate 

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

        # concatenate NLP vectors
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
        #ax_opt = ca.vertcat(*ax_opt)
        #ay_opt = ca.vertcat(*ay_opt)
        #ec_opt = ca.vertcat(*ec_opt)
    
        # ------------------------------------------------------------------------------------------------------------------
        # CREATE NLP SOLVER ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # fill nlp structure
        nlp = {'f': J, 'x': w, 'g': g}

        # solver options
        opts = {"expand": True, 
                "ipopt.max_iter": 20000,
                "ipopt.tol": 1e-7}

    

        # create solver instance
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # ------------------------------------------------------------------------------------------------------------------
        # SOLVE NLP --------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------


        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)



        if solver.stats()['return_status'] != 'Solve_Succeeded':
                print('\033[91m' + 'ERROR: Optimization did not succeed!' + '\033[0m')

        # ------------------------------------------------------------------------------------------------------------------
        # EXTRACT SOLUTION -------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # helper function to extract solution for state variables, control variables, tire forces, time
        f_sol = ca.Function('f_sol', [w], [x_opt, u_opt, dt_opt],
                                ['w'], ['x_opt', 'u_opt', 'dt_opt'])

        # extract solution
        x_opt, u_opt, dt_opt = f_sol(sol['x'])

        # solution for state variables
        x_opt = np.reshape(x_opt, (N + 1, nx))

        # solution for control variables
        u_opt = np.reshape(u_opt, (N, nu))

    
        # solution for time
        t_opt = np.hstack((0.0, np.cumsum(dt_opt)))

        print("Optimal laptime is: ", t_opt)

        # Convert numpy arrays to pandas DataFrames
        t_opt = np.reshape(t_opt, (-1, 1))
        df0 = np.concatenate((t_opt, x_opt), axis=1)
        df1 = pd.DataFrame(df0)
        df2 = pd.DataFrame(u_opt)
        df1 = pd.concat([df1, df2], axis=1)

        header = ['time', 'v', 'beta', 'omega_z', 'n', 'xi', 'wfr', 'wfl', 'wbl', 'wbr', 'delta', 'torque_drive', 'f_brake', 'gamma_x', 'gamma_y']

        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter('arrays10.xlsx', engine='xlsxwriter')

        # Write the dataframe to a excel sheet
        df1.to_excel(writer, sheet_name='Sheet1', index=False, header=header)

        # Close the Pandas Excel writer and output the excel file
        writer.close()

        return -x_opt[:-1, 3], x_opt[:-1, 0]


opt_mintime()