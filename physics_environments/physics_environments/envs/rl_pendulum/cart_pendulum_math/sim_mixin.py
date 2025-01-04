from typing import Any, Dict

import numpy as np
import pandas as pd
import sympy as smp
from scipy.integrate import solve_ivp


class CartPendulumSimulationMixin():
    """ Mixin class that provides simulation methods for Cart_Pendulum_Environment.
        Needs self.math was set by Cart_Pendulum_Math before usage.

        For non-RL simulations: use this class as provided.
        To do RL just use: angular_data  = solve_ivp(lambda t_,x_:
                                   robotic_equation(*x_, df_input.loc[np.round(t_, decimals=decimals)][0], *self.ddqf_constants['sym_to_val'].values()), # having to round some t_ indexes as scipy does not return them in 4 digits
                                   t_span=[0,t[-1]], y0=initial_values, method='RK45', t_eval=t, rtol=1e-6)
        inside the step function.
        - Update: with JAX use solver.step() using a Diffrax solver.
    """

    def full_simulation_data(self, **params : Dict[str, Any]) -> pd.DataFrame:
        r"""Legacy method:
                - wont be needed for RL
                - but can help as reference code for other types ofsimulations.

              Returns full simulation trajectory, allows only non-reactive control input.
            Allows fast parallel computation for testing.
            The control function has to be a differentiable sympy function of the x-cart-position only dependent on t.
            Define a 0 function to disable any control input.
            - The control function should of course be chosen, so that the cart will not leave the track bounds
            - This function does not implement bumping for computational efficiency; bumping is later ignored too for RL
            - Note: parallel computation of this simulation is easily possible when ignoring bumping
                --> In order to implement bumping here correctly, one needs to implement a step-wise calculation of acceleration, velocity, and position, and apply the solver for EVERY SINGLE step, not just once.
                --> The if statements along the step-wise integrating are slow too, and do not really allow flexible simulations for more than 10000 steps.
                --> Also consider using \ddot{x_c} as control function input, not x_c
         """

        assert (len(params['initial_angles'])     == self.n)
        assert (len(params['initial_velocities']) == self.n)
        assert (params['dt'] < 0.5)

        T = params['T']
        dt = params['dt'] # like 0.001
        steps = int(np.ceil(T/dt))
        s = str(dt)
        decimals = len(s[s.find('.')+1:])
        t = np.linspace(0, T, steps+1)
        t = np.round(t, decimals=decimals) # avoiding 1.250000001 s
        initial_angles     = params['initial_angles']
        initial_velocities = params['initial_velocities']
        control_func       = params['control_function']
        control_func_xc    = smp.lambdify([self.variables['t']], [control_func]) # (just for data analysis)
        control_func_dxc   = smp.lambdify([self.variables['t']], [control_func.diff(self.variables['t'], 1)]) # position to velocity (just for data analysis)
        control_func_ddxc  = smp.lambdify([self.variables['t']], [control_func.diff(self.variables['t'], 2)]) # position to acceleration
        robotic_equation   = self.math['str_to_sol_func']['ddq_f']
        initial_values = initial_angles + initial_velocities # list appending

        # calculating additional x_cart values
        if control_func != 0:
            xc_data   = np.array(control_func_xc(t))
            dxc_data  = np.array(control_func_dxc(t))
            ddxc_data = np.array(control_func_ddxc(t))
        elif control_func == 0:
            xc_data   = np.zeros((1, steps+1))
            dxc_data  = np.zeros((1, steps+1))
            ddxc_data = np.zeros((1, steps+1))

        df_input = pd.DataFrame(ddxc_data[0], index = t) # allowing for t_ indexing

        angular_data  = solve_ivp(lambda t_,x_:
                                  robotic_equation(*x_, df_input.loc[np.round(t_, decimals=decimals)][0], *self.ddqf_constants['sym_to_val'].values()), # having to round some t_ indexes as scipy does not return them in 4 digits
                                  t_span=[0,t[-1]], y0=initial_values, method='RK45', t_eval=t, rtol=1e-6)


        t_data      = t
        theta_data  = angular_data['y'][       : self.n]
        dtheta_data = angular_data['y'][self.n : self.n*2]
        ddtheta_data = np.diff(dtheta_data)/dt
        ddtheta_data = np.concatenate(([[0]]*self.n, ddtheta_data), axis=1) # [[0]]*n --> [[0], [0], [0]]

        sim_df = self.get_simdf(t_data       = t_data,
                                xc_data      = xc_data,
                                dxc_data     = dxc_data,
                                ddxc_data    = ddxc_data,
                                theta_data   = theta_data,
                                dtheta_data  = dtheta_data,
                                ddtheta_data = ddtheta_data)
        return sim_df

    def get_simdf(self,
                  t_data       : np.array,
                  xc_data      : np.array,
                  dxc_data     : np.array,
                  ddxc_data    : np.array,
                  theta_data   : np.array,
                  dtheta_data  : np.array,
                  ddtheta_data : np.array,
                 ) -> pd.DataFrame:
        """ Constructs a pd.DataFrame from a set of numpy arrays.
            The function provides mathematical column labels and also calculates the cartesian rod tip positions, required for animations, and plotting.
            - It Returns a pd.DataFrame with a suitable and minimal format for plotting and further data analysis.

            Args:
                - 7 1D/2D input arrays:
                    - time:                     t_data        (n_time_steps)
                    - cart x position:          xc_data       (n_time_steps)
                    - cart x velocity:          dxc_data      (n_time_steps)
                    - cart x acceleration:      ddxc_data     (n_time_steps)
                    - rod net angles:           theta_data    (n_rods, n_time_steps)
                    - rod angular velocities:   dtheta_data   (n_rods, n_time_steps)
                    - rod angular acceleration: ddtheta_data  (n_rods, n_time_steps)

            Output
                simdf with format:
                    - Index colum: time t
                    - Feature columns: x_c, dx_c, dd_x_c, p_x_n^r, p_y_n^r, θ_n, dθ_n, ddθ_n
        """

        str2val       = self.math['str_to_val']
        n_input       = theta_data.shape[0]
        n_input_range = np.arange(1, n_input)
        rod_lengths   = np.array([[str2val[f"l_{n_}"]] for n_ in range(1, n_input_range)]) # shape (n_rods, 1)
        pxr_summands  = -rod_lengths*np.sin(theta_data) # element-wise product
        pyr_summands  =  rod_lengths*np.cos(theta_data) # shape (n_rods, n_time_steps)
        pxr_data      = np.cumsum(pxr_summands, axis=0)
        pyr_data      = np.cumsum(pyr_summands, axis=0)

        index_name = "$$t$$"
        data_dict  = {  index_name                 : t_data,
                        "$$x_c$$"                  : xc_data,
                       r"$$\dot{x_c}$$"            : dxc_data,
                       r"$$\ddot{x_c}$$"           : ddxc_data}                                | \
                     {fr"$$p_{{x_{n_}}}^{{r}}$$"   : pxr_data[n_]     for n_ in n_input_range} | \
                     {fr"$$p_{{y_{n_}}}^{{r}}$$"   : pyr_data[n_]     for n_ in n_input_range} | \
                     {fr"$$\theta_{{{n_}}}$$"      : theta_data[n_]   for n_ in n_input_range} | \
                     {fr"$$\dot{{\theta_{n_}}}$$"  : dtheta_data[n_]  for n_ in n_input_range} | \
                     {fr"$$\ddot{{\theta_{n_}}}$$" : ddtheta_data[n_] for n_ in n_input_range}

        sim_df = pd.DataFrame(data = data_dict).set_index(index_name)

        return sim_df
