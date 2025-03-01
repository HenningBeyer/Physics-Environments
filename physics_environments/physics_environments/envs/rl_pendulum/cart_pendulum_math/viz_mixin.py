from typing import Union

import pandas as pd
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
from IPython.display import HTML, display
from plotly.graph_objs import Figure as Plotly_Figure
import plotly.express as px

from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from typing import List, Dict, Tuple
import plotly.graph_objects as go

from physics_environments.envs.rl_pendulum.cart_pendulum_math.utils_plotting import get_empty_square_animation_fig, update_figure_with_anim_widgets_and_style_buttons




class CartPendulumVisualizationMixin():
    """ Mixin class that provides visualization methods for Cart_Pendulum_Environment.
        Needs self.math was set by Cart_Pendulum_Math before usage.
        This code can be used in conjunction with a Jupyter Notebook or a Panel UI.
    """

    def math_summary(self):
        # unpacking self.math
        ## placing all naming dependencies here:
        str2func  = self.math['str_to_func']
        # str2sym   = self.math['str_to_sym']
        str2mat   = self.math['str_to_matrix']
        str2eqn   = self.math['str_to_equation']
        # t         = self.variables['t']
        x_c_func  = str2func['x_c']
        L_func    = str2func['L']
        # T_tran    = str2func['T_{trans}']
        # T_rot     = str2func['T_{rot}']
        # V         = str2func['V']

        q                 = str2mat['q']
        dq                = str2mat[r'\dot{q}']
        ddq               = str2mat[r'\ddot{q}']
        pxm_strings       = [fr"p_{{x_{n_}}}^{{m}}" for n_ in self.n_range]
        pym_strings       = [fr"p_{{y_{n_}}}^{{m}}" for n_ in self.n_range]
        T_trans_strings   = ["T_{trans_{c}}"] + [f"T_{{trans_{n_}}}" for n_ in self.n_range] # T_trans for m_c ignored for feature engineering
        T_rot_strings     = [f"T_{{rot_{n_}}}"                       for n_ in self.n_range]
        V_strings         = [f"V{n_}"                                for n_ in self.n_range]
        pxm_functions     = [str2func[str_] for str_ in pxm_strings]
        pym_functions     = [str2func[str_] for str_ in pym_strings]
        T_trans_functions = [str2func[str_] for str_ in T_trans_strings]
        T_rot_functions   = [str2func[str_] for str_ in T_rot_strings]
        V_functions       = [str2func[str_] for str_ in V_strings]
        Lode              = str2eqn['L_{ODE}']
        M                 = str2mat['M(q)']
        Lode_diff         = str2eqn['L_{ODE,diff}']
        C                 = str2mat[r'C(q, \dot{q})']
        g                 = str2mat['g(q)']
        D                 = str2mat['D']
        b                 = str2mat['b(q)']
        Lode_robotic      = str2eqn['L_{ODE,robotic}']
        theta_strings    = [fr"\theta_{n_}"   for n_ in self.n_range]
        dtheta_strings   = [fr"\dot{{\theta_{n_}}}"  for n_ in self.n_range]
        theta_sol_symbols = [smp.Symbol(str_) for str_ in (theta_strings + dtheta_strings)]
        theta_sol_matrix  = smp.Matrix(theta_sol_symbols)
        # dqq_solution_eqn  = str2eqn[r'\hat{q}']

        # q strings
        q_strings = fr"q = {smp.latex(q)}, " + fr"\dot{{q}} = {smp.latex(dq)}, " + fr"\ddot{{q}} = {smp.latex(ddq)}"

        # p_strings
        p_strings = ''
        mat = smp.Matrix([x_c_func, 0])
        p_strings += r"p_0 = " + smp.latex(mat) + ', '
        for n_ in self.n_range:
            mat = smp.Matrix([pxm_functions[n_-1], pym_functions[n_-1]])
            p_strings += fr"p_{n_} = " + smp.latex(mat) + ', '
        p_strings = p_strings[:-2]

        # all remaining strings
        ones        = smp.Matrix(np.ones(self.n+1, dtype=int))
        T_trans_mat = smp.Matrix(T_trans_functions)
        T_rot_mat   = smp.Matrix([0] + T_rot_functions)
        V_mat       = smp.Matrix([0] + V_functions)
        T_trans_string   = r"T_{trans} = " + smp.latex(T_trans_mat) + r"\cdot" + smp.latex(ones)
        T_rot_string     = r"T_{rot} = "   + smp.latex(T_rot_mat)   + r"\cdot" + smp.latex(ones)
        V_string         = r"V = "         + smp.latex(V_mat)       + r"\cdot" + smp.latex(ones)
        L_string         = r"L = (T_{trans} + T_{rot}) - V = " + smp.latex(L_func)
        Lode_string      = r"L_{ODE} = \frac{d}{dt}(\frac{\partial L}{\partial q_i}) - \frac{\partial L}{\partial q_i} = " + smp.latex(Lode)
        M_string         = r"M(q) = J_{L_{ODE}}(\ddot{q}) = " + smp.latex(M)
        Lode_diff_string = r"L_{ODE,diff} = L_{ODE} - M(q)\ddot{q} = " + smp.latex(Lode_diff)
        C_string         = r"C(q, \dot{q}) = \frac{1}{2}J_{L_{ODE,diff}}(\dot{q}) = " + smp.latex(C)
        g_string         = r"g(q) = L_{ODE,diff} - C(q, \dot{q})\dot{q} = " + smp.latex(g)
        D_string         = r"D = " + smp.latex(D)
        b_string         = r"b(q) = " + smp.latex(b)
        Lode_r_string    = r"L_{ODE,robotic} = M(q)\ddot{{q}} + C(q, \dot{{q}})\dot{{q}} + g(q) + D\dot{{q}} - b(q)\ddot{{x_c}} = " + smp.latex(Lode_robotic)
        ddq_sol_string   = r"\hat{q} = " + smp.latex(theta_sol_matrix) + " = ... =" # ' = ' + smp.latex(dqq_solution_eqn)

        html_str = f"<div style='text-align: left; line-height: 1.5'> \
                      <h4> Cart-Pendulum Dynamics </h3> \
                      <br> \
                      <blockquote> \
                        ${q_strings}$ <br><br> \
                        ${p_strings}$ <br><br> \
                        ${T_trans_string}$ <br><br> \
                        ${T_rot_string}$ <br><br> \
                        ${V_string}$ <br><br> \
                        ${L_string}$ <br><br> \
                        ${Lode_string}$ <br> \
                     </blockquote> \
                     </div> \
                     <br> \
                     <div style='text-align: left; line-height: 1.5'> \
                     <h4> Robotic Cart-Pendulum Dynamics </h3> \
                     <br> \
                     <blockquote> \
                        $M(q)\ddot{{q}} + C(q, \dot{{q}})\dot{{q}} + g(q) + D\dot{{q}} - b(q)\ddot{{x_c}} = 0$ <br><br> \
                        ${M_string}$ <br><br> \
                        ${Lode_diff_string}$ <br><br> \
                        ${C_string}$ <br><br> \
                        ${g_string}$ <br><br> \
                        ${D_string}$ <br><br> \
                        ${b_string} = -M$[:,0] <br><br> \
                        All link-to-cart-forces can be ignored when having a speed controller, \
                        hence all matrices above get their first row and column removed, and vectors get only their first row removed. <br><br> \
                        ${Lode_r_string}$ <br><br> \
                        ${ddq_sol_string}$ M.LUsolve(Lode_robotic) <br><br> \
                     </blockquote> \
                     </div> \
        "
        display(HTML(html_str))



    def get_pendulum_animation_fig_plotly(self,
                                          sim_df,
                                          frame_freq   : float = 0.02,
                                          trace_len    : int   = 500,
                                          cart_width   : float = None,
                                          track_width  : float = None) -> Plotly_Figure:
        """ This function takes the cart pendulum simulation DataFrame and creates an Plotly animation from it.
            Plotly is way faster in rendering animations in the notebook than matplotlib, and it is easily usable with Panel as animation too.

            sim_df:
                - a dataframe with at least the columns '$$x_c$$', r'$$\ddot{x_c}$$', and fr"$$\theta_{{{n_}}}$$" for all n_ in self.n_range
            frame_freq = 0.02:
                - The rendering quality is specified in 1/fps or the sampling frequency to sample from the DataFrame time series.
                - recommended: 1/50 = 0.02 for best quality, 1/32 = 0.03125 (32 fps) for clear animations; 24 ≈ 0.0415 for faster animations
                - Higher frequencies increase time to render and file size.
                - Note that higher frequencies like 0.002 cant be rendered with real-time speed, they will be rendered in slow-motion as plotly tries to render any frame without skipping
            trace_len = 500:
                - The number of trace data points being rendered per frame.
                - High numbers can slow down the animation and increase file size; set to 0 to disable traces being rendered.
            cart_width:
                - visual width of the cart in m
                - automatically defined based on the length of the first rod, if not specified explicitely
            track_width:
                - visual width of the cart track in m
                - automatically defined based on the cart_width, if not specified explicitely
        """



        def prepare_data(sim_df) -> Tuple[np.array]:
            """ This function does all the data preparation and returns rod joint position and the box coordinates. """
            def get_pxr_pyr_data() -> Tuple[np.array]:
                cart_pos = np.array([ sim_df.loc[:, '$$x_c$$'].to_numpy(),  np.zeros(len(sim_df)) ]) # rod origin

                pxr_summands = np.array([-l_list[n_-1]*np.sin(sim_df.loc[:, fr"$$\theta_{{{n_}}}$$"]) for n_ in self.n_range])
                pyr_summands = np.array([ l_list[n_-1]*np.cos(sim_df.loc[:, fr"$$\theta_{{{n_}}}$$"]) for n_ in self.n_range])

                rod_tip_xpos = np.array([np.sum(pxr_summands[:n_], axis=0) for n_ in self.n_range]) + sim_df.loc[:, '$$x_c$$'].to_numpy()
                rod_tip_ypos = np.array([np.sum(pyr_summands[:n_], axis=0) for n_ in self.n_range])

                rod_pos_x = np.vstack((cart_pos[0], rod_tip_xpos))
                rod_pos_y = np.vstack((cart_pos[1], rod_tip_ypos))
                return rod_pos_x, rod_pos_y


            def get_box_coordinate_data() -> Tuple[np.array]:
                box_x_left   = rod_pos_x[:, :] - cart_width/2
                box_x_right  = rod_pos_x[:, :] + cart_width/2
                box_coords_x = np.array([[box_x_left[0, i], box_x_right[0, i], box_x_right[0, i], box_x_left[0, i], box_x_left[0, i]]
                                        for i in range(len(sim_df))])

                box_y_down   = -cart_height/2
                box_y_up     = +cart_height/2
                box_coords_y = np.array([[box_y_down, box_y_down, box_y_up, box_y_up, box_y_down]
                                        for i in range(len(sim_df))])
                return box_coords_x, box_coords_y

            def get_arrow_magniutes() -> Tuple[np.array]:
                """ The magnitude will always be normed to [0, 1] for better visuals. """
                acc_data   = sim_df.loc[:, r'$$\ddot{x_c}$$'].to_numpy()
                max_       = np.abs(np.max(acc_data)) # simply norming all vectors with their maximum; this can only be misleading for actions --> hover the data to see the real value
                magnitudes = acc_data/max_
                return magnitudes, acc_data

            rod_pos_x, rod_pos_y       = get_pxr_pyr_data()
            box_coords_x, box_coords_y = get_box_coordinate_data()
            magnitudes, acc_data       = get_arrow_magniutes()
            data                       = (rod_pos_x, rod_pos_y, box_coords_x, box_coords_y, magnitudes, acc_data)

            return data

        def get_tracename2id_mapping() -> Dict[str, List[int]]:
            """ This is a getter function. All trace names of the figure should be mapped to their index inside the list in fig.data.

                Refer to the function get_initial_traces when needing to update this function.
            """
            return {'error_bar'        : [0],
                    'cart_box'         : [1],
                    'acc_vector'       : [2],
                    'rod_tracers'      : [3,4,5,6],
                    'rods_with_joints' : [7,8,9,10],
                }

        def get_traceid_list() -> List[int]:
            """ Simply return all indices of the function get_tracename2id_mapping as list """
            return [0,1,2,3,4,5,6,7,8,9,10]

        # Add the orginal traces to be updated in an animation:
        def init_traces(fig, data) -> None:
            """ Sets the initial figure traces of the first frame.
                Some traces will be updated, and some are constant
            """
            rod_pos_x, rod_pos_y, box_coords_x, box_coords_y, magnitudes, acc_data = data
            colors = px.colors.qualitative.Plotly

            def get_initial_traces() -> List[go.Scatter]:
                error_bar        = go.Scatter(x=[0],y=[0], hoverinfo='skip', mode='lines',
                                            error_x=dict(type='constant', value=track_width/2, color='#222244', width=8, thickness=1.8))
                cart_box         = go.Scatter(x=box_coords_x[0],
                                            y=box_coords_y[0],
                                            fill       = 'toself',
                                            opacity    = 0.8,
                                            mode       = 'lines',
                                            hoverinfo  = 'skip',
                                            line_color = 'MidNightBlue',
                                            line_width = 1.5,
                                            fillcolor  = 'rgba(175,175,175,1.0)')
                acc_vector       = go.Scatter(x=[rod_pos_x[0, 0], rod_pos_x[0, 0] + magnitudes[0]],
                                            y=[0,0],
                                            hovertemplate= f"unit value: {magnitudes[0]:.5f}<br>" +\
                                                            f"real value: {acc_data[0]:.5f}",
                                            name    = "Cart Acc.",
                                            opacity = 0.0, # the first arrow often looks bugged when its magnitude it 0.0
                                            line    = dict(color=colors[1]), # colors[1] is red
                                            marker  = dict(size=10, color=colors[1], symbol= "arrow", angleref="previous")
                                            )

                rod_tracers      = [go.Scatter(x     = [rod_pos_x[n_, 0]],
                                            y     = [rod_pos_y[n_, 0]],
                                            mode  = 'lines',
                                            name  = f"Rod {n_}",
                                            line  = dict(color=colors[n_-1], width=1.5))
                                            for n_ in self.n_range]

                rods_with_joints = [go.Scatter(x             = [rod_pos_x[n_-1, 0], rod_pos_x[n_, 0]],
                                            y             = [rod_pos_y[n_-1, 0], rod_pos_y[n_, 0]],
                                            marker_symbol = 'circle', #hoverinfo='skip',
                                            name          = f"Rod {n_}",
                                            marker        = dict(size = 10, color='Grey',
                                                                    line = dict(width = 1.5, color = 'MidNightBlue')),
                                            line          = dict(color = colors[n_-1], width = 4)
                                        ) for n_ in self.n_range[::-1]]
                initial_traces = [error_bar] + [cart_box] + [acc_vector] + rod_tracers + rods_with_joints
                return initial_traces

            initial_traces = get_initial_traces()
            fig.add_traces(initial_traces)

            ### add a timer
            timer = dict(yref='paper', x=timer_pos[0], y=timer_pos[1], ax=0, ay=0, text=f"{0:.{num_decimals}f} s") # 0 s --> 0.000 s on 3 decimals
            fig.add_annotation(timer)

        def set_frames(fig, data, sim_df_index) -> None:
            """ This function sets all the frames on a figure. These frames are used to animate the figure with a button menu. """
            rod_pos_x, rod_pos_y, box_coords_x, box_coords_y, magnitudes, acc_data = data
            colors = px.colors.qualitative.Plotly

            def get_frame_data() -> List[go.Scatter]:
                data = [[go.Scatter(x          = box_coords_x[k_],   # cart_box shape
                                    y          = box_coords_y[k_],
                                    fill       = 'toself',
                                    opacity    = 0.8,
                                    mode       = 'lines',
                                    hoverinfo  = 'skip',
                                    line_color = 'MidNightBlue',
                                    line_width = 1.5,
                                    fillcolor  = 'rgba(175,175,175,1.0)')] +\
                        [go.Scatter(x       = [rod_pos_x[0, k_], rod_pos_x[0, k_] + magnitudes[k_]],  # acc_vector
                                    y       = [0,0],
                                    hovertemplate= f"unit value: {magnitudes[k_]:.5f}<br>" +\
                                                f"real value: {acc_data[k_]:.5f}",
                                    name         = "Cart Acc.",
                                    opacity = 1.0,
                                    line=dict(color=colors[1]), # colors[1] is red
                                    marker  = dict(size=10, color=colors[1], symbol= "arrow", angleref="previous"))] +\
                        [go.Scatter(x     = rod_pos_x[n_, k_ - trace_len : k_],    # rod tracers
                                    y     = rod_pos_y[n_, k_ - trace_len : k_],
                                    name  = f"Rod {n_}",
                                    mode  = 'lines',
                                    line  = dict(color=colors[n_-1], width=1.5))
                            for n_ in self.n_range] +\
                        [go.Scatter(x             = [rod_pos_x[n_-1, k_], rod_pos_x[n_, k_]], # rods and rod joints
                                    y             = [rod_pos_y[n_-1, k_], rod_pos_y[n_, k_]],
                                    marker_symbol = 'circle',
                                    name          = f"Rod {n_}",
                                    marker        = dict(size = 10, color ='Grey', line = dict(width = 1.5, color = 'MidNightBlue')),
                                    line          = dict(color = colors[n_-1], width = 4))
                            for n_ in self.n_range[::-1]]

                    for k_ in np.int64(sim_df_index/mean_df_frequency)
                ]
                return data

            def get_step_layouts() -> List[go.Layout]:
                """ Just updating the timer in here.
                    Other layout updates can be defined here too.
                """
                layout = [go.Layout(annotations=[dict(x=timer_pos[0], y=timer_pos[1], ax=0, ay=0, yref='paper',
                                    text=f"{full_sim_df_idx[k_]:.{num_decimals}f} s")]) # 0.5 s --> 0.500 s on 3 decimals
                        #.add_annotation(
                                        #,
                                    # yref='paper', x=0, y=-0.2)
                            for k_ in np.int64(sim_df_index/mean_df_frequency)]
                return layout

            frame_data   = get_frame_data()
            frame_layout = get_step_layouts()
            frame_names  = [f'frame{full_sim_df_idx[k_]}' for k_ in np.int64(sim_df_index/mean_df_frequency)]
            frames       = [go.Frame(data   = frame_data[i],
                                     traces = get_traceid_list(), # update traces specified in initial_traces with target indizes (i.e. [error_bar] + [cart_box] + [acc_vector] + rod_tracers + rods_with_joints.)
                                     layout = frame_layout[i],
                                     name   = frame_names[i]) for i in range(len(frame_names))]
            fig.update(frames=frames)  # All updates should be defined with one Frame object list and fig.update, else this produces an incorrect animation


        ### Constants
        l_list            = [self.constants['str_to_val'][f'l_{n_}'] for n_ in self.n_range]
        total_rod_len     = sum(l_list)

        if cart_width is None:           # if not specified, cart_width will be 0.75 time first rod length
            l1            = self.constants['str_to_val'][f'l_{1}']
            cart_width    = 0.75*l1
        if track_width is None:
            track_width   = 3.5*cart_width # if not specified, track_width will be 3.5x the cart_width



        cart_height       = cart_width*2/5
        fig_x_lim         = 0.5*track_width + total_rod_len
        fig_y_lim         = fig_x_lim

        mean_df_frequency = np.abs(np.mean(sim_df.index[:-1] - sim_df.index[1:])) # should return the same as self.dt; this is kept flexible for irregular data
        step_frequency    = int(frame_freq/mean_df_frequency) # render every 20 steps for 0.02/0.001

        str_              = str(mean_df_frequency)   # 0.001
        str_              = str_[str_.index('.')+1:] # 001
        num_decimals      = len(str_)                # --> 3
        fig_margin        = dict(t=30,l=30, r=30, b=175)
        timer_pos         = (0, 0.05)
        ###

        ### Main Script
        fig                  = get_empty_square_animation_fig(title       = 'Cart-Pendulum Animation',
                                                              xaxis_title = r"$\large{x \text{ (m)}}$",
                                                              yaxis_title = r"$\large{y \text{ (m)}}$",
                                                              x_lims      = (-fig_x_lim, fig_x_lim),
                                                              y_lims      = (-fig_y_lim, fig_y_lim),
                                                              margin      = fig_margin)
        data                 = prepare_data(sim_df=sim_df)
        init_traces(fig, data)
        sim_df_sample        = sim_df.iloc[::step_frequency]
        full_sim_df_idx      = sim_df.index
        set_frames(fig, data, sim_df_index=sim_df_sample.index)
        update_figure_with_anim_widgets_and_style_buttons(fig                  = fig,
                                                          frame_freq           = frame_freq,
                                                          tracename2id_mapping = get_tracename2id_mapping(),
                                                          toggled_trace_names  = ['acc_vector', 'rod_tracers'],
                                                          transition_duration  = 0,
                                                          slider_x             = 0.094,
                                                          player_x             = 1.04,
                                                          dropdown_pos         = (1.25,1.0),
                                                          fig_margin           = fig_margin)
        ###

        return fig



    def get_state_fig(self,
                      pxr_t0        : np.array,
                      pyr_t0        : np.array,
                      x_c_t0        : float,
                      ddx_c_t0      : float,
                      fig_size      : float = 8
                      ) -> Tuple[Figure, Axes]:

        """ Function that returns the initial figure for an animation of animate_simulation_data()
            This first frame renders the cart pendulum with its rods, cart, and acceleration in a square figure.
            This function is also relevant, if one just wants to render the current state of the environment.

            Inputs:
                - pxr_t0: 1D np.array of shape (n_rods); cartesian rod x-positions
                - pyr_t0: 1D np.array of shape (n_rods); cartesian rod y-positions
                - x_c_t0: float with the current cart x-position
                - ddx_c_t0: float with the current cart x-acceleration

            Outputs:
                fig, ax, plus also all components needed
        """
        pxr_dict   = {fr"$p_{{x_{n_}}}^{{r}}$" : val_ for n_, val_ in enumerate(pxr_t0)}
        pyr_dict   = {fr"$p_{{y_{n_}}}^{{r}}$" : val_ for n_, val_ in enumerate(pyr_t0)}
        x_c_dict   = {'$$x_c$$'        : x_c_t0,
                     r'$$\ddot{x_c}$$' : ddx_c_t0}
        data_dict  = pxr_dict | pyr_dict | x_c_dict
        sim_dft0   = pd.DataFrame(data_dict)
        fig, ax    = self.animate_simulation_data(sim_df                  = sim_dft0,
                                                  trace_len               = 0,
                                                  fig_size                = fig_size,
                                                  _return_only_first_plot = True)

        return fig, ax


    def animate_simulation_data(self,
                                sim_df                  : pd.DataFrame,
                                shown_traces            : List[bool],
                                fps                     : int   = 30,
                                trace_len               : int   = 450,
                                fig_size                : int   = 8,
                                show_cart_acc_vector    : bool  = True,
                                cart_width              : float = None,
                                track_width             : float = None,
                                _return_only_first_plot : bool  = False,
                               ) -> Union[matplotlib.animation.FuncAnimation,
                                          Tuple[Figure, Axes]]:
        """ Note:
                This is a legacy function which still works very well, but the animate_simulation_data_plotly animation
                looks a bit better and renders much, much faster, and with more functionalities.

            Args:
              sim_df:
                - pd.DataFrame with specific input columns and a time index '$$t$$'.
                - input columns must include at least ['$$x_c$$, '$$\ddot{x_c}$$'] and also should contain ['$$\theta_{n_}$$']
                    - If [fr"$$p_{{x_{n_}}}^{{r}}$$"] + [fr"$$p_{{y_{n_}}}^{{r}}$$"] should not be present already, they will be calculated. (this will need ['$$\theta_{n_}$$'] columns to be present)
                    - Else ['$$\theta_{n_}$$'] would be optional
              shown_traces:
                - A list of booleans to toggle the visibility of each trace.
                - The list has to contain n bools like [True, False, False, False] for n=4.
              fps:
                - frames per second
                - higher number corresponds to smoother animations, but way longer rendering times
                - also can be seen as the number of datapoints rendered per second.
              trace_len:
                - The number of trace data points being rendered per frame.
                - High numbers can slow down the animation and increase file size; set to 0 to disable traces being rendered .
              fig_size:
                - The size of the square figure
              show_cart_acc_vector:
                - wheter to show the normalized acceleration vector of the cart
             cart_width:
                - visual width of the cart in m
                - automatically defined based on the length of the first rod, if not specified explicitely
             track_width:
                - visual width of the cart track in m
                - automatically defined based on the cart_width, if not specified explicitely
              _return_only_first_plot:
                - Is a lazy implementation to return just the first frame as plot, for plotting a single state
                - Should only be set to True, if the first plot should be returned instead of the animtion

            Returns:
                - matplotlib.animation.FuncAnimation
                --> render it in a notebook via IPython.display.HTML(anim_output.to_jshtml())
        """

        def calc_pxr_pyr_data(x_c_data    : np.array,  # shape (t,)
                              theta_data  : np.array,  # shape (n, t)
                              ) -> Tuple[np.array, np.array]:
            """ calculate the cartesian rod tip position data, if missing """
            rod_lengths  = np.array([self.constants['str_to_val'][f"l_{n_}"] for n_ in self.n_range])
            pxr_summands = -rod_lengths*np.sin(theta_data) # element-wise product
            pyr_summands =  rod_lengths*np.cos(theta_data)
            pxr_data = np.cumsum(pxr_summands, axis=0) + x_c_data
            pyr_data = np.cumsum(pyr_summands, axis=0)
            return pxr_data, pyr_data

        # General animation parameters
        ## fps*t_total (t_total is in s!) --> frames = number of update function calls
        frames = fps*sim_df.index[-1] # fps*total time (in s) = num frames
        data_step_size = int(np.floor(len(sim_df)/frames))

        # Data Processing
        cols = [col_.replace('$$', '$') for col_ in sim_df.columns] # plt hates plotting '$$' strings
        sim_df = sim_df.set_axis(labels=cols, axis=1, copy=False)

        x_c_data   = np.array(sim_df['$x_c$'])
        ddx_c_data = np.array(sim_df[r'$\ddot{x_c}$'])
        theta_data = np.array([sim_df[fr'$\theta_{{{n_}}}$']    for n_ in self.n_range])

        pxr_pyr_is_present = fr"$p_{{x_{1}}}^{{r}}$" in sim_df.columns
        if pxr_pyr_is_present:
            pxr_data   = np.array([sim_df[fr"$p_{{x_{n_}}}^{{r}}$"] for n_ in self.n_range])
            pyr_data   = np.array([sim_df[fr"$p_{{y_{n_}}}^{{r}}$"] for n_ in self.n_range])
        else:
            pxr_data, pyr_data = calc_pxr_pyr_data(x_c_data   = x_c_data,
                                                   theta_data = theta_data)


        # Constants
        l_sum           = sum([self.constants['str_to_val'][f'l_{n_}'] for n_ in self.n_range])
        cart_pos0       = (x_c_data[0]  , 0)
        cart_acc0       = (ddx_c_data[0], 0)
        cart_height     = cart_width*2/5
        if cart_width is None:           # if not specified, cart_width will be 0.75 time first rod length
            l1          = self.constants['str_to_val'][f'l_{1}']
            cart_width  = 0.75*l1
        if track_width is None:
            track_width = 3.5*cart_width # if not specified, track_width will be 3.5x the cart_width

        # Style Parameters/Constants
        fig_square_size = 8 # 800x800 pixel base size
        fig_x_lim = 0.5*track_width+l_sum
        fig_y_lim = fig_x_lim
        x_lim_scaling = 0.9
        y_lim_scaling = 0.9
        lw_scaling = fig_size/8
        colors = ['#D03030','#5050D0','#30C030','#E09020','#30A0A0','#8060C0','#C050C0']*3 # or use cmap = plt.get_cmap('Set1'); color_1 = cmap. which looks a bit more unpleasant



        # Figure
        fig, ax = plt.subplots(figsize=( fig_square_size*x_lim_scaling, fig_square_size*y_lim_scaling) )

        # Main plot adjustments
        fig.tight_layout()
        ax.axis('scaled') # center to (0,0)
        ax.grid(True, which='major', alpha=0.75)
        ax.grid(True, which='minor', alpha=0.5)
        ax.set_xlim((-fig_x_lim*x_lim_scaling, fig_x_lim*x_lim_scaling))
        ax.set_ylim((-fig_y_lim*y_lim_scaling, fig_y_lim*y_lim_scaling))
        ax.use_sticky_edges = True

        # Setting labels and titles
        ax.set_title('Cart-Pendulum Animation')
        ax.set_ylabel('$y$' + ' / ' '$m$')
        ax.set_xlabel('$x$' + ' / ' '$m$')

        # Adjusting tick style of axes (now handled by the new plot styling)
        #         major_tick_offset = 0.25
        #         num_minor_ticks = 2
        #         ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(major_tick_offset))
        #         ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(major_tick_offset/(num_minor_ticks+1)))
        #         ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(major_tick_offset))
        #         ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(major_tick_offset/(num_minor_ticks+1)))

        # Defining the ground axis
        ax.errorbar(0, 0, xerr=track_width/2, fmt=' ', color='k', capsize=6,capthick=1.5, lw=1.25, zorder=5) # ground line + boundary ticks
        ax.errorbar(0, 0, xerr=track_width/4, fmt=' ', color='k', capsize=4,capthick=1.1, lw=0.25, zorder=5) # half-way ticks
        ax.errorbar(0, 0, xerr=0,             fmt=' ', color='k', capsize=4,capthick=1.1, lw=0.25, zorder=5) # starting tick

        # Defining the timer
        str_  = f'{sim_df.index[0]} s'
        timer = plt.text(-0.005, min(-0.15,-cart_height/2), s=str_, fontsize=12)

        # Defining the cart box
        rectangle = plt.Rectangle(xy=(0-cart_width/2,-cart_height/2), width=cart_width, height=cart_height,
                                      animated = True, antialiased=True, snap=True,
                                      linewidth=1.0, edgecolor='#202020', facecolor='#808080', zorder=2, alpha=1)

        # Defining rods
        rod0   =        [plt.Line2D((cart_pos0[0], pxr_data[n_-1, 0]), (cart_pos0[1], pyr_data[n_-1, 0]), c=colors[0], animated=True, antialiased=True, clip_on=True, solid_capstyle='round', lw=3.5*lw_scaling, alpha=1,  zorder=6)
                         for n_ in [1]]
        rods   = rod0 + [plt.Line2D(pxr_data[n_-1:n_+1, 0],             pyr_data[n_-1:n_+1, 0],           c=colors[n_-1+1], animated=True, antialiased=True, clip_on=True, solid_capstyle='round', lw=3.5*lw_scaling, alpha=1,  zorder=6)
                         for n_ in self.n_range[0:-1]]

        # Defining rod joints
        joint0 =          [plt.Line2D((cart_pos0[0], pxr_data[n_-1, 0]), (cart_pos0[1], pyr_data[n_-1, 0]), marker='o', animated=True, antialiased=True, clip_on=True, solid_capstyle='round',
                           markeredgecolor='#202020',  markerfacecolor='#B5B5B5', markeredgewidth=0.75, markersize=8*lw_scaling, lw=0, alpha=1, markevery=(2),  zorder=6)
                           for n_ in [1]]
        joints = joint0 + [plt.Line2D(pxr_data[n_-1:n_+1, 0],             pyr_data[n_-1:n_+1, 0],           marker='o', animated=True, antialiased=True, clip_on=True, solid_capstyle='round',
                           markeredgecolor='#202020',  markerfacecolor='#B5B5B5', markeredgewidth=0.75, markersize=8*lw_scaling, lw=0, alpha=1, markevery=(2),  zorder=6)
                           for n_ in self.n_plus_1_range[0:-1]]

        # Defining tracing lines of joints
        tracer = [plt.Line2D((0,0),(0,0), c=colors[n_-1], animated=True, antialiased=True, clip_on=True, solid_capstyle='round', lw=1.1, alpha=1,  zorder=3)
                  if shown_traces[n_-1] is True
                  else plt.Line2D((0,0),(0,0), alpha=0) # still need an object to return in update_fig
                  for n_ in self.n_range]

        # Defining cart acceleration vector
        if show_cart_acc_vector:
            a_vector = ax.quiver(cart_pos0[0], cart_pos0[1], cart_acc0[0], cart_acc0[1], scale=1/0.335, width=0.0035, color='#C03030', alpha=1, zorder=6)
        else:
            a_vector = ax.quiver(cart_pos0[0], cart_pos0[1], cart_acc0[0], cart_acc0[1], alpha=0) # still need an object to return in update_fig

        # Building the plot
        for n_ in self.n_range:
            ax.add_line(rods[n_-1])
            ax.add_line(tracer[n_-1])
        for n_ in self.n_plus_1_range:
            ax.add_line(joints[n_-1])
        ax.add_patch(rectangle)


        def update_fig(frame_num): # animation function: this is called sequentially when using animation.FuncAnimation()
            """ This function is called n times (n = frames) to pick and visualize m data
                samples (m = interval) for each second.
            """
            i_ = frame_num
            timer_str = f'{np.round(sim_df.index[i_*data_step_size],2)} s'
            cart_pos = (  x_c_data[i_*data_step_size], 0)
            cart_acc = (ddx_c_data[i_*data_step_size], 0)
            rod_pos = np.array([[pxr_data[n_-1, i_*data_step_size],
                                 pyr_data[n_-1, i_*data_step_size]]
                                 for n_ in self.n_range]) # shape

            tracer_traj = np.array([[pxr_data[n_-1, max(0, i_*data_step_size - trace_len) : i_*data_step_size],
                                     pyr_data[n_-1, max(0, i_*data_step_size - trace_len) : i_*data_step_size]]
                                    for n_ in self.n_range]) # shape (n, 2, trace_len)

            # Update Timer
            timer.set_text(timer_str)

            # Update rods
            rod_start_pos = cart_pos
            for n_ in self.n_range:
                rod_end_pos = rod_pos[n_-1]
                rods[n_-1].set_data([rod_start_pos[0], rod_end_pos[0]], [rod_start_pos[1], rod_end_pos[1]])
                rod_start_pos = rod_end_pos

            # Update Joints
            ## Joints are also Line2D objects
            ## Joints are only a start position marker
            rod_start_pos = cart_pos
            for n_ in self.n_range:
                rod_end_pos = rod_pos[n_-1]
                joints[n_-1].set_data([rod_start_pos[0], rod_end_pos[0]], [rod_start_pos[1], rod_end_pos[1]])
                rod_start_pos = rod_end_pos
            joints[-1].set_data([rod_start_pos[0], rod_end_pos[0]], [rod_start_pos[1], rod_end_pos[1]])

            # Update Tracers
            for n_ in self.n_range:
                if shown_traces[n_-1] is True:
                    tracer[n_-1].set_data(tracer_traj[n_-1, 0, :], tracer_traj[n_-1, 1, :])

            # Update Cart
            rectangle.set_xy( (cart_pos[0]-cart_width/2, cart_pos[1]-cart_height/2)
                            )

            # Update acceleration Vector
            if show_cart_acc_vector:
                a_vector.set_UVC(cart_acc[0], cart_acc[1])
                a_vector.set_offsets([cart_pos[0],cart_pos[1]])

            return (timer, *rods, *joints, *tracer, rectangle, a_vector)


        if _return_only_first_plot is False:
            plt.close() # output from fig, ax rendering
            return animation.FuncAnimation(fig, update_fig, frames=int(np.floor(frames)), interval=1/fps*1000, blit=True)
        else:
            return fig, ax # lazy implementation to return a single state figure instead of an animation.

    def simulation_dataset_plot(self, sim_df) -> None:
        """ Plots a Simulation 2x2 Summary given a DataFrame with the format from Cart_Pendulum_Simulation_Mixin"""
        fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained', figsize=(14,8))
        fig.suptitle(f'Simulation Data Visualisation ($n$ = {self.n})', fontsize='large')

        cols = [col_.replace('$$', '$') for col_ in sim_df.columns] # plt hates plotting '$$' strings
        sim_df = sim_df.set_axis(labels=cols, axis=1, copy=False)

        cols_00 = [fr'$\theta_{{{n_}}}$' for n_ in self.n_range] + [fr'$\dot{{\theta_{n_}}}$' for n_ in self.n_range]
        cols_01 = ['$x_c$', r'$\dot{x_c}$', r'$\ddot{x_c}$']
        cols_10 = [fr'$p_{{x_{n_}}}^{{r}}$' for n_ in self.n_range]
        cols_11 = [fr'$p_{{y_{n_}}}^{{r}}$' for n_ in self.n_range]

        data_00 = sim_df[cols_00]
        data_01 = sim_df[cols_01]
        data_10 = sim_df[cols_10]
        data_11 = sim_df[cols_11]

        for arr_ in axs:
            for ax in arr_:
                ax.grid()
                ax.set_xlabel(r'$t / s$')
                ax.axhline(y=0, color='k', linestyle='--', linewidth=1.1)
                ax.set_xlim(sim_df.index[0], sim_df.index[-1])
                #ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2)) (now handled by the new plot styling)
                #ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

        ax = axs[0,0]
        data_00.plot(ax=ax, grid=True)
        ax.set_title(r'$\theta$ Features')
        ax.set_ylabel(r'$\theta_n, \dot{theta_n}$ / $rad, rad/s$')

        ax = axs[0, 1]
        data_01.plot(ax=ax, grid=True)
        ax.set_title(r'$x_c$ Features')
        ax.set_ylabel(r'$x_c, \dot{x_c}, \ddot{x_c}$ / $m,m/s, m/s^2$')

        ax = axs[1, 0]
        data_10.plot(ax=ax, grid=True)
        ax.set_title(r'$p_{x_{n}}^{r}$ Features')
        ax.set_ylabel(r'$p_{x_{n}}^{r}$ / $m$')

        ax = axs[1, 1]
        data_11.plot(ax=ax, grid=True)
        ax.set_title(r'$p_{y_{n}}^{r}$ Features')
        ax.set_ylabel(r'$p_{y_{n}}^{r}$ / $m$')

        plt.show()

    def trajectory_plot(self,
                        sim_df : pd.DataFrame,
                        cart_width   : float = None,
                        track_width  : float = None) -> None:

        colors = ['#D03030','#5050D0','#30C030','#E09020','#30A0A0','#8060C0','#C050C0']*3

        cols = [col_.replace('$$', '$') for col_ in sim_df.columns] # plt hates plotting '$$' strings
        sim_df = sim_df.set_axis(labels=cols, axis=1, copy=False)

        # Constants
        total_rod_len   = sum([self.constants['str_to_val'][f'l_{n_}'] for n_ in self.n_range])
        if cart_width is None:           # if not specified, cart_width will be 0.75 time first rod length
            l1            = self.constants['str_to_val'][f'l_{1}']
            cart_width    = 0.75*l1
        if track_width is None:
            track_width   = 3.5*cart_width # if not specified, track_width will be 3.5x the cart_width
        fig_square_size = 8
        fig_x_lim = 0.5*track_width + total_rod_len
        fig_y_lim = fig_x_lim
        x_lim_scaling = 0.9
        y_lim_scaling = 0.9

        fig, ax = plt.subplots(layout='constrained', figsize=(fig_square_size*x_lim_scaling,fig_square_size*y_lim_scaling))
        fig.suptitle(f'Trajectory Visualization ($n$ = {self.n})')
        ax.grid(True, which='major', alpha=0.65)
        ax.grid(True, which='minor', alpha=0.4)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.8)
        ax.axvline(x=(track_width-cart_width)/2, color='k', linestyle='--', linewidth=1, alpha=0.8)
        ax.axvline(x=-(track_width-cart_width)/2, color='k', linestyle='--', linewidth=1, alpha=0.8)
        ax.axis('scaled') # center to (0,0)
        ax.set_xlim((-fig_x_lim*x_lim_scaling, fig_x_lim*x_lim_scaling))
        ax.set_ylim((-fig_y_lim*y_lim_scaling, fig_y_lim*y_lim_scaling))
        ax.scatter(x=(track_width-cart_width)/2, y=0, color='k', marker='x')
        ax.scatter(x=-(track_width-cart_width)/2, y=0, color='k', marker='x')
        ax.scatter(x=0, y=0, color='k', marker='x')

        # Adjusting tick style of axes (now handled by the new plot styling)
        #         major_tick_offset = 0.25
        #         num_minor_ticks = 2
        #         ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(major_tick_offset))
        #         ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(major_tick_offset/(num_minor_ticks+1)))
        #         ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(major_tick_offset))
        #         ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(major_tick_offset/(num_minor_ticks+1)))

        # Defining Ground Axis
        ax.errorbar(0, 0, xerr=track_width/2, fmt=' ', color='k', capsize=5,capthick=1.0, lw=0.8, zorder=5) # ground line + boundary ticks
        ax.errorbar(0, 0, xerr=track_width/4, fmt=' ', color='k', capsize=3,capthick=1.0, lw=0.25, zorder=5) # half-way ticks
        ax.errorbar(0, 0, xerr=0,             fmt=' ', color='k', capsize=3,capthick=1.0, lw=0.25, zorder=5) # starting tick

        x_cols = [fr'$p_{{x_{n_}}}^{{r}}$' for n_ in self.n_range]
        y_cols = [fr'$p_{{y_{n_}}}^{{r}}$' for n_ in self.n_range]
        for x_col,y_col, i in zip(x_cols, y_cols, range(self.n)):
            sim_df.plot(x=x_col, y=y_col, ax=ax, grid=True, color=colors[i], lw=1)

        ax.set_ylabel('$y$' + ' / ' '$m$')
        ax.set_xlabel('$x$' + ' / ' '$m$') # avoid having x label overwritten by plotting

        plt.show()







