from typing import Sequence

import chex
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import pandas as pd
from IPython.display import HTML
from typing import Dict, List

from physics_environments.envs.rl_pendulum.cart_pendulum_math  import CartPendulumPhysicsEnvironment
from physics_environments.envs.rl_pendulum.types               import Observation, State, Constants, StateUtils, ObservationUtils
from physics_environments.envs.rl_pendulum.utils_plotting      import get_pxr_pyr_data
from physics_environments.envs.rl_pendulum.cart_pendulum_math.utils_plotting import \
    plot_matplotlib_with_ieee_style, get_empty_ts_plot_figure_plotly, save_plotly_animation_to_mp4

from jumanji.viewer import Viewer

class PendulumViewer(Viewer[State]):
    """ This Viewer class provides its viewing capabilites explicitely for the notebook.
        - The PendulumViewer mostly interfaces the visualization methods from Cart_Pendulum_Physics_Environment,
          and further adds some more for analyzing the State and Observation sequence data of the RL_Cart_Pendulum_Environment.

        Note that most of the methods are meant to be interfaced by RL_Cart_Pendulum_Environment.viewer.render_math_summary
          for isolating any specific rendering method into the viewer attribute.
        This design choice was taken as scientific environments/simulations will in general need a lot of dedicated visualization/animation code
          that might be separated to redice clutter.
    """

    def __init__(self,
                 physics_env : CartPendulumPhysicsEnvironment,
                 constants   : Constants):
        self.physics_env       = physics_env
        self.constants         = constants
        self.state_utils       = StateUtils()
        self.observation_utils = ObservationUtils()

    def render_state_matplotlib(self, state: State, save_fig=False, save_filename='plot.png') -> None:
        """ Renders the pendulum for a single state/animation frame. """
        pxr_data, pyr_data = get_pxr_pyr_data(rod_lengths = self.constants.rod_lengths,
                                              thetas      = state.thetas,
                                              x_c_data    = state.s_x)
        fig, ax = self.physics_env.get_state_fig(pxr_t0     = pxr_data,
                                                 pyr_t0     = pyr_data,
                                                 x_c_t0     = state.s_x,
                                                 ddx_c_t0   = state.a_x)
        plt.show()

    def render_state_plotly(self, state: State, save_fig=False, save_filename='plot.png') -> None:
        """ Renders the pendulum for a single state/animation frame. """
        sim_df = self.get_state_dataframe([state]) # sim_df has length of 1


        sim_df = pd.concat([sim_df, sim_df], ignore_index=True)
        sim_df.index = [0, self.constants.dt] # need at least two df entries so that frequency calculations are correct in the below function

        # lazily calling this to render an entire animation, but from one state; it works just too well
        plotly_fig   = self.physics_env.get_pendulum_animation_fig_plotly(sim_df     = sim_df,
                                                                                  frame_freq = self.constants.dt,
                                                                                  trace_len  = 0)
        plotly_fig.layout.frames      = [{}] # removing all frames
        plotly_fig.layout.updatemenus = [{}] # remove any buttons and widgets to keep the plot only

        plotly_fig.show()

    def plot_episode_data_matplotlib(self,
                                     df            : pd.DataFrame,
                                     col_names     : List[str]     = None,
                                     save_fig      : bool          = False,
                                     save_params   : dict          = {'save_filename' : 'matplotlib_plot.png', # or 'matplotlib_plot.svg'
                                                                      'dpi'           : 250}
                                     ) -> None:
        """ Plots the observation features (or any other data) for an episode.
            Plots are also plotted in a more scientific style, using this function.

            Args:
                df:        Any pd.DataFrame works here. The index will be plotted as x-axes data.
                col_names: The list of features/features columns to plot.
                save_fig:  Wheter to also save the rendered figure using save_params.
                save_params:
                    - default value: {'save_filename'  : 'plotly_plot.html',   # or 'matplotlib_plot.svg'
                                      'dpi'            : 250}                  # used for .png

        """
        if col_names is None:
            col_names = df.columns # if no columns specified, plot the entire Dataframe.

        # Saving a .png or .svg:
        plot_matplotlib_with_ieee_style(df            = df,
                                        col_names     = col_names,
                                        title         = 'Episode Feature Visualization',
                                        xaxis_title   = r"$t \text{ (s)}$",
                                        yaxis_title   = r'$\text{Features (a.u.)}$',
                                        save_fig      = save_fig,
                                        save_filename = save_params['save_filename'],
                                        dpi           = save_params['dpi'])


    def plot_episode_data_plotly(self,
                                 df             : pd.DataFrame,
                                 col_names      : List[str] = None,
                                 save_fig       : bool      = False,
                                 save_params    : dict      = {'save_filename'  : 'plotly_plot.html', # or plotly_plot.png, or plotly_plot.svg
                                                               'height'         : None,               # in pixels; only used for .png, and .svg
                                                               'width'          : None,               # in pixels; only used for .png, and .svg
                                                               'scale'          : 1.0}                # in pixels; only used for .png, and .svg
                                 ) -> None:
        """ Plots the observation features (or any other data) for an episode.

            Plots plotted with this function receive a more readable scientific style for analyzing the many features for RL control systems.
            Plotly makes the chart also interactive, and non-pixelated.

            Args:
                df:        Any pd.DataFrame works here. The index will be plotted as x-axes data.
                col_names: The list of features/features columns to plot.
                save_fig:  Wheter to also save the rendered figure using save_params.
                save_params:
                    - default value: {'save_filename'  : 'plotly_plot.html', # or plotly_plot.png, or plotly_plot.svg
                                      'height'         : None,               # in pixels; only used for .png, and .svg
                                      'width'          : None,               # in pixels; only used for .png, and .svg
                                      'scale'          : 1.0}                # in pixels; only used for .png, and .svg
        """
        if col_names is None:
            col_names = df.columns # if no columns specified, plot the entire Dataframe.

        plotly_fig = get_empty_ts_plot_figure_plotly(title       = 'Episode Feature Visualization',
                                                     xaxis_title = r"$\large{t \text{ (s)}}$",
                                                     yaxis_title = r'$\large{\text{Features (a.u.)}}$')

        plotly_fig = get_empty_ts_plot_figure_plotly()
        lines      = [go.Scatter(x=df.index, y=df[col_], name=col_, mode='lines') for col_ in col_names]
        plotly_fig.add_traces(lines)

        save_filename = save_params['save_filename']
        if save_fig:
            if '.html' in save_filename: # Saving interactive HTML
                plotly_fig.write_html(file            = save_filename,
                                      include_mathjax = 'cdn', # enable LaTeX rendering in the saved file
                                      auto_play       = True)
            elif ('.png' in save_filename) or\
                 ('.svg' in save_filename):
                plotly_fig.write_image(file    = save_filename,
                                       engine  = "auto", # needs either "kaleido" or "orca engine installed"
                                       width   = save_params['width'],
                                       height  = save_params['height'],
                                       scale   = save_params['scale'])
            else:
                raise ValueError(f"Only file formats of .html and .svg are recommended for saving plotly figures. However, {save_params['save_filename']} indicated a different file type.")

        plotly_fig.show()

    def animate_episode_plotly(self,
                               sim_df               : pd.DataFrame,
                               frame_freq           : float          = 0.02,
                               trace_len            : int            = 300,
                               save_anim            : bool           = False,
                               render_animation     : bool           = True,
                               save_params          : dict           = {'save_filename'  : 'plotly_animation.html', # or 'plotly_animation.mp4'
                                                                        'height'         : None,                    # in pixels; only used for .mp4
                                                                        'width'          : None,                    # in pixels; only used for .mp4
                                                                        'scale'          : 1.0},                    # in pixels; only used for .mp4
                               ) -> None:

        """ Returns an interactive Plotly animation to render it in a notebook. Plotly can render significantly faster than matplotlib, and looks better.

            Plotly always renders each frame without frameskip to not miss out one frame for interactivity reasons.
            This may play the animation in slow-motion for very data-intensive animations, i.e. with high fps and trace len, compared to matplotlib.

            Args:
                sim_df
                    - pd.DataFrame with required input columns and a time index '$$t$$'.
                    - input columns must include at least ['$$\theta_{n_}$$'] + ['$$x_c$$, '$$\ddot{x_c}$$']
                frame_freq
                    - Default:0.02, #
                    - frame_freq in s;
                    - renders a frame each 20 ms = 1/50 (50 fps); this will affect rendering time AND disk space of animation
                    - recommended:
                        - 1/50 = 0.02    (50 fps) for highest-quality animations
                        - 1/32 = 0.03125 (32 fps) for high-quality animations
                        - 1/20 = 0.05    (20 fps) for good-quality animations
                    - higher frequencies like 0.002 cant be rendered with real-time speed, they will be rendered in slow-motion as plotly tries to render any frame without skipping
                trace_len
                    - The length of the plotted rod tracers following the rod-tip positions
                    - This is big contributer to high save space and slower rendering time
                    - recommended:
                        - 300 for almost all cases; not too long, not too short.
                        - 0   for disabled rod tracers
                save_anim:
                    - Wheter to save the animation or not
                    - On save_anim = True, the animation will be saved as .mp4, according to save_filename.
                render_animation:
                    - whether to render the animation in a notebook or not.
                    - render_animation = False, can be advantageous to avoid exeedingly long rendering times, when you only want to save the animation.
                save_params:
                    - default values:
                        save_params  = {'save_filename'  : 'plotly_animation.html', # or 'plotly_animation.mp4'
                                        'height'         : None,                    # in pixels; only used for .mp4
                                        'width'          : None,                    # in pixels; only used for .mp4
                                        'scale'          : 1.0},                    # in pixels; only used for .mp4
                    - save_filename can specify the save location, file-type and file-name
                    - height, width, and scale correspond to the png image sizes, used to render the .mp4 animation
        """

        plotly_fig =  self.physics_env.get_pendulum_animation_fig_plotly(sim_df     = sim_df,
                                                                         frame_freq = 0.02,
                                                                         trace_len  = 300)

        assert (save_anim or render_animation), ('Both save_anim and render_animation were set to false, so this function does nothing')
        if save_anim:
            if '.html' in save_params['save_filename']:  # Saving interactive HTML
                plotly_fig.write_html(file            = save_params['save_filename'],
                                      include_mathjax = 'cdn', # enable LaTeX rendering in the saved file
                                      auto_play       = True)

            elif '.mp4' in save_params['save_filename']: # Saving a non-interactive .mp4 animation
                fps = int(1/frame_freq)
                save_params.update({'fps' : fps}) # additionally include fps into params

                save_plotly_animation_to_mp4(plotly_fig  = plotly_fig,
                                             save_params = save_params)
            else:
                raise ValueError(f"Only file formats of .html and .mp4 are recommended for saving plotly animations. However, {save_params['save_filename']} indicated a different file type.")

        if render_animation:
            plotly_fig.show()

    def animate_episode_matplotlib(self,
                                   sim_df               : pd.DataFrame,
                                   shown_traces         : List[bool],
                                   fps                  : int  = 30,
                                   trace_len            : int  = 450,
                                   fig_size             : int  = 8,
                                   show_cart_acc_vector : bool = True,
                                   save_anim            : bool = False,
                                   render_animation     : bool = True,
                                   save_params          : dict = {'save_filename' : 'matplotlib_animation.gif', # or 'matplotlib_animation.mp4'
                                                                  'dpi'           : 250},
                                   ) -> None:
        """ Renders a matplotlib animation in a notebook. Matplotlib can take 2-3 mins to render one animation, because mpl_anim.to_jshtml() takes that long.
            Additionally, if save_anim == True, the animation is saved upon calling this function.

            Matplotlib may only have the small advantage, that the animation time exactly matches the real-time by default.

            Args:
              sim_df:
                - pd.DataFrame with specific input columns and a time index '$$t$$'.
                - input columns must include at least ['$$\theta_{n_}$$'] + ['$$x_c$$, '$$\ddot{x_c}$$']
              shown_traces:
                - A list of booleans to toggle the visibility of each trace.
                - The list has to contain n bools like [True, False, False, False] for n=4.
              fps:
                - frames per second
                - higher number corresponds to smoother animations, but way longer rendering times
                - also can be seen as the number of datapoints rendered per second.
                - higher fps means faster animation play-speed for the viewer
              trace_len:
                - The number of trace data points being rendered per frame.
                - High numbers can slow down the animation and increase file size; set to 0 to disable traces being rendered .
              fig_size:
                - The size of the square figure
              show_cart_acc_vector:
                - wheter to show the normalized acceleration vector of the cart
              save_anim:
                - Wheter to save the animation or not
                - On save_anim = True, the animation will be saved as .mp4, according to save_filename.
              render_animation:
                - whether to render the animation in a notebook or not.
                - render_animation = False, can be advantageous to avoid long exeedingly long rendering times (especially for matplotlib).
              save_params:
                - Default value:    dict = {'save_filename' : 'matplotlib_animation.gif',    # or 'matplotlib_animation.mp4'
                                            'dpi'           : 250},
        """

        mpl_anim = self.physics_env.animate_simulation_data(sim_df               = sim_df,
                                                            shown_traces         = shown_traces,
                                                            fps                  = fps,
                                                            trace_len            = trace_len,
                                                            fig_size             = fig_size,
                                                            show_cart_acc_vector = show_cart_acc_vector)

        if save_anim:
            if '.gif' in save_params['save_filename'] or\
               '.mp4' in save_params['save_filename']:
                mpl_anim.save(filename = save_params['save_filename'],
                              writer   = 'ffmpeg',
                              fps      = fps,
                              dpi      = save_params['dpi'])
            else:
                raise ValueError(f"Only file formats of .gif and .mp4 are recommended for saving matplotlib animations. However, {save_params['save_filename']} indicated a different file type.")

        if render_animation:
            HTML(mpl_anim.to_jshtml())



    # The following methods were outsourced to types.py for better mainainability:

    ## Methods mostly for debugging, potentially usable for the user:
    def get_state_dataframe(self, states: Sequence[State], drop_debug_data=True) -> pd.DataFrame:
        """ Returns a pd.DataFrame with a suitable format for plotting and further data analysis from the recorded state sequence. """
        return self.state_utils.get_state_dataframe(states = states)

    def get_observation_dataframe(self, n : int, observations: Sequence[Observation]) -> pd.DataFrame:
        """ Returns a pd.DataFrame with a suitable format for plotting and further data analysis from the recorded observation sequence. """
        return self.observation_utils.get_observation_dataframe(n = n, observations = observations)

    def _get_raw_state_data_dict(self, states: Sequence[State]) -> Dict[str, chex.Array]:
        """ Converts a state sequence to a slightly more usable data_dictionary of state sequences. """
        return self.state_utils.get_raw_state_data_dict(states = states)

    def _get_raw_observation_data_dict(self, observations: Sequence[Observation]) -> Dict[str, chex.Array]:
        """ Converts an observation sequence to a slightly more usable data_dictionary of observation sequences. """
        return self.state_utils.get_raw_observation_data_dict(observations = observations)

    def _get_prepared_state_data_dict(self, states: Sequence[State]) -> Dict[str, chex.Array]:
        """ Converts a state sequence to a slightly more usable data_dictionary of state sequences.
            It splits up feature groups, and gives proper mathematical symbols to the each state feature. """
        return self.state_utils.get_prepared_state_data_dict(states = states)

    def _get_prepared_observation_data_dict(self, observations: Sequence[Observation]) -> Dict[str, chex.Array]:
        """ Converts an observation sequence to a slightly more usable data_dictionary of observation sequences.
            It splits up feature groups, and gives proper mathematical symbols to each observation feature. """
        return self.state_utils.get_prepared_observation_data_dict(observations = observations)


    # Defining the mandatory abstract methods of Viewer
    def close(self):
        plt.close

    def animate(self):
        raise NotImplementedError('Refer to the alternative animation functions of the viewer instead!')

    def render(self):
        raise NotImplementedError('Refer to the alternative render functions of the viewer instead!')




















