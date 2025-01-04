
from typing import Literal, Tuple
from functools import cached_property

import chex
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import ODETerm, Tsit5
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition, truncation

from physics_environments.envs.rl_pendulum.cart_pendulum_math.env import CartPendulumPhysicsEnvironment
from physics_environments.envs.rl_pendulum.cart_pendulum_math.param_manager import CartPendulumParameterManager
from physics_environments.envs.rl_pendulum.cart_pendulum_math.robotic_eq_solutions import robotic_eq_solution_dict

from physics_environments.envs.rl_pendulum.done import TerminateDoneFn, TruncateDoneFn
from physics_environments.envs.rl_pendulum.generator import BenchmarkGenerator, DefaultGenerator, UniformGenerator
from physics_environments.envs.rl_pendulum.reward import DefaultRewardFn
from physics_environments.envs.rl_pendulum.types import Constants, MathObjects, Observation, State, RLCartPendulumEnvironmentParams
from physics_environments.envs.rl_pendulum.utils import get_base_features, get_rod2cart_distance_features, get_theta_features
from physics_environments.envs.rl_pendulum.viewer import PendulumViewer


class RLCartPendulumEnvironment(Environment[State, specs.MultiDiscreteArray, Observation]):
    """
    ** RLCartPendulumEnvironment **
        A simple and flexible n-rodded RL cart pendulum environment.

    Environment Information:
    - action:
        - 1 continuous action a_x (x accelearation of the cart)
    - reward:
        - mean of all rod cosines (1 reward if all rods standing, 0 reward if all rods hanging)
        - additional -15 penalty if bumping into the environment boundaries
    - episode termination:
        - if at least one cart bumps into the environment boundaries
        - and on reaching the termination time t_episode
    - state:
        - Information for used for env.step, and to calculate observations.
        - only should have impact on the environment dynamics
    - observation:
        - what the agent will see and the agent's acting is based on
        - will be fully calculated from the environment state
        - 4 base features + 9n theta features + 9n rod2cart distance features = 18n + 4 features
    """

    def __init__(self,
                 params : RLCartPendulumEnvironmentParams = RLCartPendulumEnvironmentParams()
                 ):
        if params._init_phyiscs_env_from_file is True:
            # loading a physics environment object initialized in the past; can be convenient
            physics_env  = CartPendulumParameterManager.load_env_obj_from_file(filename=params._physics_env_init_filename)
        else:
            # initializing a new object, can take long for n > 3 specified in params!
            physics_env  = CartPendulumPhysicsEnvironment(params          = params.physics_env_params,
                                                          initialize_math = params._initialize_env_math # _initialize_env_math is always False for usage; Skips the compute-intensive and optional symbolic calculations.
                                                          )

        n = physics_env.constants['str_to_val']['n']
        rod_lengths   = [physics_env.constants['str_to_val'][f"l_{n_}"] for n_ in physics_env.n_range] # like [0.15, 0.15, 0.15] for n = 3
        ode_constants = jnp.array(list(physics_env.ddqf_constants['str_to_val'].values()))

        self.constants       = Constants(n                     = n,
                                         dt                    = params.dt,
                                         t_episode             = params.t_episode,
                                         max_cart_acceleration = params.max_cart_acceleration,
                                         bump_x                = (params.track_width-params.cart_width)/2,
                                         cart_width            = params.cart_width,
                                         track_width           = params.track_width,
                                         rod_lengths           = jnp.array(rod_lengths),
                                         ode_constants         = ode_constants)

        ode_function  = robotic_eq_solution_dict[f'n={n}']  # function f(t, y, args) # --> t is not used; y = [theta_1, dtheta_1, ddx_c]; args = (g, r_1, m_1, mu_1, I_1) for n=1
        self.math_objects    = MathObjects(solver              = Tsit5(),
                                           ode_term            = ODETerm(ode_function))

        self.reward_function = DefaultRewardFn(bumping_corner_distance = self.constants.bump_x,
                                               reward_exponent = params.reward_exponent)

        if params.training_generator_type == "DefaultGenerator":
            self.training_generator = DefaultGenerator(constants = self.constants)
        elif params.training_generator_type == "UniformGenerator":
            self.training_generator = UniformGenerator(constants = self.constants)
        else:
            raise ValueError(f"The training_generator_type type {params.training_generator_type} might have been misspecified.")

        self.benchmark_generator = BenchmarkGenerator(constants = self.constants)
        self.terminate_done_fn   = TerminateDoneFn(t_episode = self.constants.t_episode)
        self.truncate_done_fn    = TruncateDoneFn(bump_x = self.constants.bump_x)

        self.viewer              = PendulumViewer(physics_env = physics_env,
                                                  constants   = self.constants)
        super().__init__()


    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """ Resets the environment. """
        state              = self.training_generator(key)
        # Initialize solver
        solver_state       = self.math_objects.solver.init(terms=self.math_objects.ode_term, y0=state.y_solver, t0=state.t, t1=state.t+self.constants.dt, args=self.constants.ode_constants)
        state.solver_state = solver_state

        observation = self._state_to_observation(state=state)
        timestep    = restart(observation=observation) # returns class TimeStep set with initial observation and reward
        return state, timestep

    def reset_to_benchmark(self, key : chex.PRNGKey, benchmark_id : int) -> Tuple[State, ]:
        """ Resets the environment for a Benchmark. """
        state              = self.benchmark_generator(key, benchmark_id)

        # Initialize solver
        solver_state       = self.math_objects.solver.init(terms=self.math_objects.ode_term, y0=state.y_solver, t0=state.t, t1=state.t+self.constants.dt, args=self.constants.ode_constants)
        state.solver_state = solver_state
        state.key          = key

        observation = self._state_to_observation(state=state)
        timestep    = restart(observation=observation) # returns class TimeStep set with initial observation and reward
        return state, timestep



    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """ Run one timestep of the environment's dynamics. action is expected to have shape (1,). """
        s_x, v_x, action  = state.s_x, state.v_x, jnp.squeeze(action) # squeeze reduces action of shape (1,) to shape (), but also allows action input shape ()

        # step calculation

        a_x = jnp.clip(action, min=-self.constants.max_cart_acceleration, max=self.constants.max_cart_acceleration)  # clip action
        a_x = jnp.nan_to_num(a_x, nan=0.0)
        v_x = v_x + a_x*self.constants.dt
        s_x = s_x + v_x*self.constants.dt + a_x*self.constants.dt*self.constants.dt/2
        t_next     = state.t+self.constants.dt
        step_count = state.step_count + 1

        state.y_solver   = state.y_solver.at[..., -1].set(a_x) # give the action to the solver

        y_solver, _, _, solver_state, _ = self.math_objects.solver.step(terms=self.math_objects.ode_term,
                                                                        t0=state.t, t1=t_next, y0=state.y_solver, args=self.constants.ode_constants,
                                                                        solver_state=state.solver_state, made_jump=False)
        #jax.debug.print("{y_solver}, {state}", y_solver=y_solver, state=state)
        n = self.constants.n
        thetas       = y_solver[...,   : n    ] # y_solver: [thetas, dthetas, a_x]
        dthetas      = y_solver[..., n : n + n]
        ddthetas = (dthetas - state.dthetas)/self.constants.dt # works fine like an actual derivative function

        reset_cart = self.truncate_done_fn(state = state)

        s_x = jax.lax.cond( # set s_x to zero if it bumped into a corner
            reset_cart,
            lambda x: jnp.array(0.0),
            lambda x: s_x,
            jnp.array([]) # no operands needed
        )

        next_state = State(t            = t_next,
                           step_count   = step_count,
                           key          = state.key,
                           solver_state = solver_state,
                           y_solver     = y_solver,
                           thetas       = thetas,
                           dthetas      = dthetas,
                           ddthetas     = ddthetas,
                           s_x          = s_x,
                           v_x          = v_x,
                           a_x          = a_x)

        reward          = self.reward_function(state)

        next_observation = self._state_to_observation(state=next_state)
        terminate = self.terminate_done_fn(state = next_state)  # terminate is the naturally intended episode ending, or when the goal was reached by the agent
        truncate  = False#self.truncate_done_fn(state = state)  # truncate is the abrupt episode ending if the agent could not reach the goal, or if it produced a disallowed state
                # --> Instead of truncating, the s_x position is set to 0 again, which does not cause many issues with the current features which are mostly relative rod position features to the cart and rod angles.


        next_timestep = jax.lax.switch(
            terminate + 2 * truncate,
            [lambda rew, obs, shape: transition( reward=rew, observation=obs, shape=shape),  # 0: !terminate and !truncate
             lambda rew, obs, shape: termination(reward=rew, observation=obs, shape=shape),  # 1:  terminate and !truncate
             lambda rew, obs, shape: truncation( reward=rew, observation=obs, shape=shape),  # 2: !terminate and  truncate
             lambda rew, obs, shape: termination(reward=rew, observation=obs, shape=shape)], # 3:  terminate and  truncate
            reward,
            next_observation,
            (), # shape parameter of rewards; needed for MARL envs
        )

        return next_state, next_timestep

    def _state_to_observation(self, state: State) -> Observation:
        """ Takes a state from env.step and converts it to an observation, i.e. the agent input.

            It is possible to feature engineering
        """

        # base_features               = jnp.array([s_x, v_x, a_x, d_corner])
        base_features                 = get_base_features(bump_x = self.constants.bump_x,
                                                          s_x    = state.s_x,
                                                          v_x    = state.v_x,
                                                          a_x    = state.a_x)

        # theta_features              = (thetas, dthetas, ddthetas, sin_thetas, dsin_thetas, ddsin_thetas, cos_thetas, dcos_thetas, ddcos_thetas)
        theta_features                = get_theta_features(thetas   = state.thetas,
                                                           dthetas  = state.dthetas,
                                                           ddthetas = state.ddthetas)

        # rod2cart_distance_features  = (out_r2cx, d_out_r2cx, dd_out_r2cx, out_r2cy, d_out_r2cy, dd_out_r2cy, out_r2cd, d_out_r2cd, dd_out_r2cd)
        rod2cart_distance_features    = get_rod2cart_distance_features(rod_lengths = self.constants.rod_lengths,
                                                                       thetas      = state.thetas,
                                                                       dthetas     = state.dthetas,
                                                                       ddthetas    = state.ddthetas)

        agent_inputs_global  = jnp.concatenate([base_features.flatten(), theta_features.flatten(), rod2cart_distance_features.flatten()]) # (4 + 9n + 9n features)
        return Observation(agent_inputs_global = agent_inputs_global)


    def render(self, state: State) -> None:
        raise NotImplementedError("Refer to all the visualization methods via RL_Cart_Pendulum_Environment.viewer!")

    def close(self) -> None:
        pass # no extra cleanup necessary for closing this environment

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:

        # keeping a mostly permissive definition for flexibility:
        agent_inputs_global = specs.BoundedArray(
            shape=(4 + 9*self.constants.n + 9*self.constants.n, ), # refer to _state_to_observation to get the shape
            dtype=jnp.float32,
            minimum=-jnp.inf,
            maximum=jnp.inf,
            name="agent_inputs_global",
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_inputs_global=agent_inputs_global
        )

    @cached_property
    def action_spec(self) -> specs.Array:
        return specs.BoundedArray(
                shape=(1,),
                name="cart acceleration",
                minimum=-self.constants.max_cart_acceleration,
                maximum=self.constants.max_cart_acceleration,
                dtype=jnp.float32,
            )

    def __repr__(self):
        return self.__doc__
