from typing import TYPE_CHECKING, Dict, NamedTuple, Sequence, Literal

import chex
import numpy as np
import pandas as pd
from diffrax import AbstractSolver, AbstractTerm
from typing_extensions import TypeAlias

from dataclasses import field
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

from physics_environments.envs.rl_pendulum.cart_pendulum_math.types import CartPendulumPhysicsEnvironmentParams
from physics_environments.types import ParamsBaseMixin

_SolverState: TypeAlias = None



@dataclass
class RLCartPendulumEnvironmentParams(ParamsBaseMixin):
    """ Adds additional Parameters to the RLCartPendulumEnvironmentParams

        Copy-Paste Template:
            rl_env_params = RLCartPendulumEnvironmentParams(physics_env_params          = cart_pendulum_params,
                                                            training_generator_type     = "DefaultGenerator",
                                                            dt                          = 0.010,
                                                            t_episode                   = 10.00,
                                                            max_cart_acceleration       = 25.00,
                                                            cart_width                  = 0.250,
                                                            track_width                 = 0.800
                                                            )

        Note:
            - physics_env_params does not need to be specified, if you load the physics env
              with _init_phyiscs_env_from_file and _physics_env_init_filename

        Parameter Definition:
                physics_env_params          : CartPendulumPhysicsEnvironmentParams \
                                            = field(default_factory=CartPendulumPhysicsEnvironmentParams)
                training_generator_type     : Literal["DefaultGenerator", "UniformGenerator"] \
                                            = "DefaultGenerator"                   --> the string of the class responsible for initializing/generating the first state of an episode
                dt                          : float = 0.010                        --> step time in s
                t_episode                   : float = 10.00                        --> episode time in s
                max_cart_acceleration       : float = 25.00                        --> maximum accleration the agent is allowed to make as an action; (in m/s²)
                cart_width                  : float = 0.250                        --> the cart width in m; has impact on the episode termination, reward penalty events, and vizualization
                track_width                 : float = 0.800                        --> the track width in m; has impact on the episode termination, reward penalty events, and vizualization
                reward_exponent             : float = 1.0                          --> higher values scale the reward such that rod andgles different to the standing position are rewarded significantly less; thus promotes high accuracy objectives;
                                                                                       --> formula is reward = reward_orig**reward_exponent where reward_orig has value ranges of [0, 1]

                _init_phyiscs_env_from_file : bool = False,                        --> May not change, debugging; wheter to load the physics environment from a .pkl file, instead initializing it again (can avoid long initialization times)
                _physics_env_init_filename  = str  = 'pendulum_obj_filename.pkl',  --> May not change, debugging; the file name to initialize the physics_env onject from.
                _initialize_env_math        : bool  = False                        --> May not change, debugging; When False, skips the mostly optional (and compute-intensive) symbolic calculations
    """

    physics_env_params          : CartPendulumPhysicsEnvironmentParams \
                                = field(default_factory=CartPendulumPhysicsEnvironmentParams)
    training_generator_type     : str \
                                = "DefaultGenerator"                  # the string of the class responsible for initializing/generating the first state of an episode
                                                                      # Literal["DefaultGenerator", "UniformGenerator"] --> removed this typing as it causes problems with omegaconfig
    dt                          : float = 0.010                       # step time in s
    t_episode                   : float = 10.00                       # episode time in s
    max_cart_acceleration       : float = 25.00                       # maximum accleration the agent is allowed to make as an action; (in m/s²)
    cart_width                  : float = 0.250                       # the cart width in m; has impact on the episode termination, reward penalty events, and vizualization
    track_width                 : float = 0.800                       # the track width in m; has impact on the episode termination, reward penalty events, and vizualization
    reward_exponent             : float = 1.0                         # higher values scale the reward such that rod andgles different to the standing position are rewarded significantly less; thus promotes high accuracy objectives;
                                                                        # formula is reward = reward_orig**reward_exponent where reward_orig has value ranges of [0, 1]

    _init_phyiscs_env_from_file : bool  = False                       # May not change, debugging; wheter to load the physics environment from a .pkl file, instead initializing it again (can avoid long initialization times)
    _physics_env_init_filename  : str   = 'pendulum_obj_filename.pkl' # May not change, debugging; the file name to initialize the physics_env onject from.
    _initialize_env_math        : bool  = False                       # May not change, debugging; When False, skips the mostly optional (and compute-intensive) symbolic calculations

@dataclass
class Constants:
    """ A class of constants used for all episodes. """
    n                     : int
    dt                    : float
    t_episode             : float
    max_cart_acceleration : float
    bump_x                : float
    cart_width            : float
    track_width           : float
    rod_lengths           : chex.Array
    ode_constants         : chex.Array

@dataclass
class MathObjects:
    """ A set of math objects and constants used for the environment. """
    solver      : AbstractSolver
    ode_term    : AbstractTerm

@dataclass
class State:
    """ All the necessary information to calculate agent observations and the next environment State wthin an environment step.
        Also contains some state variables needed for logging, terminating, etc. """
    t            : chex.Numeric # step time
    step_count   : chex.Numeric # step num
    key          : chex.PRNGKey # Jax.random.key(seed=42)
    solver_state : _SolverState # solver state
    y_solver     : chex.Array   # jnp.concatenate([thetas, dthetas, ddx_c])
    thetas       : chex.Array   # rod net angles
    dthetas      : chex.Array   # angular velocities
    ddthetas     : chex.Array   # angular accelerations
    s_x          : chex.Numeric # cart pos
    v_x          : chex.Numeric # cart velocity
    a_x          : chex.Numeric # cart acceleration = the clipped action

class Observation(NamedTuple):
    """ All the state features an agent receives. This is calculated from the step """
    agent_inputs_global  : chex.Array # --> shape (4 + 9n + 9n)

class StateUtils:
    """ A collection of data-processing methods strictly tied to the implementation of Observation.
        Hence, it is directly placed here to directly allow change, if the Observation class should change.

        It's currently heavily used within PendulumViewer in viewer.py
    """

    @staticmethod
    def get_raw_state_data_dict(states: Sequence[State]) -> Dict[str, chex.Array]:
        """ Converts a list ob states into a more managable dictionary of state sequences.
            This data_dict is suitable for conversion into a pd.DataFrame, or can serve for debugging.

            get_prepared_state_data_dict use this function's output to return a more user-friendly dataframe-format.

            Input:
                [state1, state2, ..., statex]

            Output:
                {'t'          : np.array([0.23, ..., -4.21]),
                 'step_count' : np.array([1.21, ..., -1.41]),
                  ...
                 'ddtheta'    : np.array([[0.20, ..., +2.21], [1.21, ..., -1.41], ... [1.15, ... -0.71]]),
                 ...
                 'a_x'        : np.array([-0.03, ..., +3.33])}
                 for n = 4
        """

        sample_state        = states[0]
        raw_state_data_keys = sample_state.keys() # --> like ['t', 'step_count', 'solver_state', 'y_solver', 's_x', 'v_x', 'a_x', 'theta', 'dtheta', 'ddtheta']
        raw_state_data_dict = {key_ : np.array([getattr(state__, key_) for state__ in states], dtype=object) for key_ in raw_state_data_keys} # --> {'t' : np.array, ..., 'ddtheta' : np.array}
                                                                                                                                # dtype=object to allow inhomogenous np.arrays

        return raw_state_data_dict

    def get_prepared_state_data_dict(self,
                                     states: Sequence[State]) -> Dict[str, chex.Array]:
        """ A extention of the function get_raw_state_data_dict.
            This function also assigns each mathmatical feature a LaTeX symbol, while splitting up feature groups into separate features:
              - Special features, like images, a solver_state or data matrices will remain compacted.
              - Feature groups like theta_features will be split up into separate features

            The output is intended to be as managable as possible for later calculations in DataFrames.

            Input:
                [state1, state2, ..., statex]

            Output:
                {'$$t$$'            : np.array([ 0.23, ..., -4.21]),
                 'step_count'       : np.array([ 1.21, ..., -1.41]),
                  ...
                 fr'$$\dot{x_c}$$'  : np.array([ 0.20, ..., +2.21]),
                 fr'$$\ddot{x_c}$$' : np.array([-0.03, ..., +3.33])}
                 for n = 4
        """  # noqa: W605

        # Implementations specific to State
        state_data_keys  = list(State.__dataclass_fields__.keys()) # --> ['t', 'step_count', 'solver_state', 'y_solver', 'thetas', 'dthetas', 'ddthetas', 's_x', 'v_x', 'a_x']
        t_, step_count_, key_, solver_state_, y_solver_, theta_, dtheta_, ddtheta_, xc_, dxc_, ddxc_  \
            = (np.array([getattr(state_, key) for state_ in states], dtype=object) for key in state_data_keys) # dtype=object to allow inhomogenous np.arrays

        n       = theta_.shape[1]
        n_range = np.arange(1, n+1)

        prepared_state_data_dict = {
            '$$t$$'          : t_,
            'step_count'     : step_count_,
            'key'            : list(key_),           # pd.DataFrame does not like 2D+ arrays as field entries, but lists work
            'solver_state'   : list(solver_state_),
            'y_solver'       : list(y_solver_),
            } |\
            {fr"$$\theta_{{{n_}}}$$"      : theta_[  :, n_-1] for n_ in n_range} | \
            {fr"$$\dot{{\theta_{n_}}}$$"  : dtheta_[ :, n_-1] for n_ in n_range} | \
            {fr"$$\ddot{{\theta_{n_}}}$$" : ddtheta_[:, n_-1] for n_ in n_range} | \
            {'$$x_c$$'        : xc_,
            r'$$\dot{x_c}$$'  : dxc_,
            r'$$\ddot{x_c}$$' : ddxc_}

        return prepared_state_data_dict


    def get_state_dataframe(self,
                            states: Sequence[State],
                            drop_debug_data = True) -> pd.DataFrame:
        """
            Inputs:
                drop_debug_data:
                    - If True, only keep the data interesting for data analysis, i.e. not the object data of solvers, etc.
                    - Else, keeps all state data for debugging, etc.
        """
        data_dict = self.get_prepared_state_data_dict(states = states)
        if drop_debug_data is True:
            for key_ in ['step_count', 'solver_state', 'key', 'y_solver']:
                data_dict.pop(key_)


        df        = pd.DataFrame(data = data_dict).set_index('$$t$$')

        return df


class ObservationUtils:
    """ A collection of data-processing methods strictly tied to the implementation of Observation.
        Hence, it is directly placed here to directly allow change, if the Observation class should change.

        It's currently heavily used within PendulumViewer in viewer.py
    """

    @staticmethod
    def get_raw_observation_data_dict(observations: Sequence[Observation]) -> Dict[str, chex.Array]:
        """ Converts a list ob observations into a more managable dictionary of observation sequences.
            This data_dict is suitable for conversion into a pd.DataFrame, or can serve for debugging.

            get_prepared_obervation_data_dict use this function's output to return a more user-friendly dataframe-format.

            Input:
                [observation1, observation2, ..., observationx]

            Output:
                {0  : np.array([[0.20, ..., +2.21], [1.21, ..., -1.41], ... [1.15, ... -0.71]])}   # just agent_inputs_global
                 for n = 4
        """

        sample_obs        = observations[0]
        raw_obs_data_keys = sample_obs.keys() # --> like ['key1', 'key2', 'key3'] or just ['agent_inputs_global'] in this implementation
        raw_obs_data_dict = {key_ : np.array([getattr(obs_, key_) for obs_ in observations], dtype=object) for key_ in raw_obs_data_keys} # --> {'agent_inputs_global' : np.array}
                                                                                                                                          # dtype=object to allow inhomogenous np.arrays

        return raw_obs_data_dict

    def get_prepared_observation_data_dict(self,
                                           n: int,
                                           observations: Sequence[Observation]) -> Dict[str, chex.Array]:
        r""" A extention of the function get_raw_observation_data_dict.
            This function also assigns each mathmatical feature a LaTeX symbol, while splitting up feature groups into separate features:
              - Special features, like images, or data matrices will remain compacted.
              - Feature groups like theta_features, base_features will be split up into separate features

            The output is intended to be as managable as possible for later calculations in DataFrames.

            Input:
                [observation1, observation2, ..., observationx]

            Output:
                {fr'$$x_c$$'                    : np.array([ 0.23, ..., -4.21]),
                 fr'$$\dot{{x_c}}$$'            : np.array([ 1.21, ..., -1.41]),
                            ...                 :                  ...
                 fr'$$\ddot{{d_{{r_{3}-c}}}}$$' : np.array([ 0.20, ..., +2.21]),
                 fr'$$\ddot{{d_{{r_{4}-c}}}}$$' : np.array([-0.03, ..., +3.33])}
                 for n = 4
        """

        # Implementations specific to Observation AND to env._state_to_observation
        obs_data_keys        = ['agent_inputs_global']
        agent_inputs_global  = (np.array([getattr(obs_, key) for obs_ in observations], dtype=object) for key in obs_data_keys) # dtype=object to allow inhomogenous np.arrays

        i2s_mapping = self.get_agent_inputs_index2str_mapping(n = n)
        i_          = 0
        out_dict    = {}

        # Unpacking agent_inputs_global with i2s_mapping, to fill out_dict
        for item_ in agent_inputs_global:
            if i2s_mapping[str(i_)] == 'IGNORE_FEATURE': # allows to exclude features from unpacking, which can be handy for 64x64 image inputs, that should be treated as one feature.
                i_ += 1
                continue
            if len(item_.shape) == 1:
                out_dict.update({i_ : item_})
                i_ += 1
            elif len(item_.shape) == 2:
                """ Assuming an array of shape (n, num_steps)"""
                for sub_array_ in item_:
                    out_dict.update({i_ : sub_array_})
                    i_ += 1
            else:
                raise NotImplementedError('No implementations for 3D arrays and arrays of higher dimensions yet.')

        prepared_state_data_dict = out_dict
        return prepared_state_data_dict

    @staticmethod
    def get_agent_inputs_index2str_mapping(n : int) -> Dict[int, str]:
        r""" Utility function for returning the mathematical symbols for all the features in types.Observation that is set by env._state_to_observation.

            These symbols are used as column names for DataFrames and provide a lot of clarity later on for plotting.
            This clarity becomes very important for analyzing 50+ Observation states.

            The indices of agent_inputs_global are mapped to their mathematical strings in here.
            The function has to be updated when changing _state_to_observation or any function in utils.py.

            Notes:
                - The oder of these string labels has to be the same as set in env._state_to_observation
                - If, for example, image features of 64x64 features should not be split up into single_features, one would have to implement dummy features,
                   and sk

            Input:
                n

            Output:
                - Dependent on agent_input, which is set in env.state_to_observation
                {0 : fr'$$x_c$$',
                 1 : fr'$$\dot{{x_c}}$$',
                  ...
                 38 : fr'$$\ddot{{d_{{r_{3}-c}}}}$$',
                 39 : fr'$$\ddot{{d_{{r_{4}-c}}}}$$'}
                 for n = 4
        """
        n_range = np.arange(1, n+1)

        # Base Features
        base_feature_symbols             = [ r'$$x_c$$',
                                             r'$$\dot{{x_c}}$$',
                                             r'$$\ddot{{x_c}}$$',
                                             r'$$d_{{corner}}$$']
        # Example Ignore Feature
        # one_ignored_feature            = ['IGNORE_FEATURE'] # the 5th feature in agent_inputs_global could be an image; this would be ignored, and not unpacked, when uncommented.

        # Theta Features
        theta_feature_symbols            = [fr'$$\theta_{{{n_}}}$$'                for n_ in n_range]
        dtheta_feature_symbols           = [fr'$$\dot{{\theta_{{{n_}}}}}$$'        for n_ in n_range]
        ddtheta_feature_symbols          = [fr'$$\ddot{{\theta_{{{n_}}}}}$$'       for n_ in n_range]

        sin_theta_feature_symbols        = [fr'$$sin(\theta_{{{n_}}})$$'           for n_ in n_range]
        dsin_theta_feature_symbols       = [fr'$$\dot{{sin(\theta_{{{n_}}})}}$$'   for n_ in n_range]
        ddsin_theta_feature_symbols      = [fr'$$\ddot{{sin(\theta_{{{n_}}})}}$$'  for n_ in n_range]

        cos_theta_feature_symbols        = [fr'$$cos(\theta_{{{n_}}})$$'           for n_ in n_range]
        dcos_theta_feature_symbols       = [fr'$$\dot{{cos(\theta_{{{n_}}})}}$$'   for n_ in n_range]
        ddcos_theta_feature_symbols      = [fr'$$\ddot{{cos(\theta_{{{n_}}})}}$$'  for n_ in n_range]

        # Rod-to-Cart Distance Features
        rod2cart_xdist_feature_symbols   = [fr'$$d^x_{{r_{n_}-c}}$$'               for n_ in n_range]
        drod2cart_xdist_feature_symbols  = [fr'$$\dot{{d^x_{{r_{n_}-c}}}}$$'       for n_ in n_range]
        ddrod2cart_xdist_feature_symbols = [fr'$$\ddot{{d^x_{{r_{n_}-c}}}}$$'      for n_ in n_range]

        rod2cart_ydist_feature_symbols   = [fr'$$d^y_{{r_{n_}-c}}$$'               for n_ in n_range]
        drod2cart_ydist_feature_symbols  = [fr'$$\dot{{d^y_{{r_{n_}-c}}}}$$'       for n_ in n_range]
        ddrod2cart_ydist_feature_symbols = [fr'$$\ddot{{d^y_{{r_{n_}-c}}}}$$'      for n_ in n_range]

        rod2cart_dist_feature_symbols    = [fr'$$d_{{r_{n_}-c}}$$'                 for n_ in n_range]
        drod2cart_dist_feature_symbols   = [fr'$$\dot{{d_{{r_{n_}-c}}}}$$'         for n_ in n_range]
        ddrod2cart_dist_feature_symbols  = [fr'$$\ddot{{d_{{r_{n_}-c}}}}$$'        for n_ in n_range]

        all_symbols_in_order =  np.concatenate([base_feature_symbols,
                                                theta_feature_symbols,          dtheta_feature_symbols,          ddtheta_feature_symbols,
                                                sin_theta_feature_symbols,      dsin_theta_feature_symbols,      ddsin_theta_feature_symbols,
                                                cos_theta_feature_symbols,      dcos_theta_feature_symbols,      ddcos_theta_feature_symbols,
                                                rod2cart_xdist_feature_symbols, drod2cart_xdist_feature_symbols, ddrod2cart_xdist_feature_symbols,
                                                rod2cart_ydist_feature_symbols, drod2cart_ydist_feature_symbols, ddrod2cart_ydist_feature_symbols,
                                                rod2cart_dist_feature_symbols,  drod2cart_dist_feature_symbols,  ddrod2cart_dist_feature_symbols])

        agent_inputs_idx2sym_dict = {ind_ : sym_ for ind_, sym_ in enumerate(all_symbols_in_order)}

        return agent_inputs_idx2sym_dict

    def get_observation_dataframe(self,
                                  n : int,
                                  observations: Sequence[Observation]) -> pd.DataFrame:

        data_dict = self.get_prepared_observation_data_dict(n = n, observations = observations)
        df        = pd.DataFrame(data = data_dict).set_index('$$t$$')
        return df
