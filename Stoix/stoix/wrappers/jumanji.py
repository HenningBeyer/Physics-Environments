from typing import Tuple

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.specs import Array, MultiDiscreteArray, Spec
from jumanji.types import TimeStep
from jumanji.wrappers import MultiToSingleWrapper, Wrapper

from stoix.base_types import Observation
from stoix.wrappers.transforms import MultiDiscreteToDiscrete

from functools import cached_property # added line hbeyer


class JumanjiWrapper(Wrapper):
    def __init__(
        self,
        env: Environment,
        observation_attribute: str,
        multi_agent: bool = False,
    ) -> None:
        if isinstance(env.action_spec, MultiDiscreteArray): # fixed bug: action_spec() --> action_spec; hbeyer
            env = MultiDiscreteToDiscrete(env)

        if multi_agent:
            env = MultiToSingleWrapper(env)

        self._env = env
        self._observation_attribute = observation_attribute

        if isinstance(env.action_spec, Array):            # added line hbeyer; jumanji never had continuous envs, so Stoix forgot to implement this.
            self._num_actions = self.action_spec().shape[0] # added line hbeyer; assuming flat action shapes
        else:
            self._num_actions = self.action_spec().num_values

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        if hasattr(timestep.observation, "action_mask"):
            legal_action_mask = timestep.observation.action_mask.astype(float)
        else:
            legal_action_mask = jnp.ones((self._num_actions,), dtype=float)
        if self._observation_attribute:
            agent_view = timestep.observation._asdict()[self._observation_attribute].astype(float)
        else:
            agent_view = timestep.observation
        obs = Observation(agent_view, legal_action_mask, state.step_count)
        timestep_extras = timestep.extras
        if not timestep_extras:
            timestep_extras = {}
        timestep = timestep.replace(
            observation=obs,
            extras=timestep_extras,
        )
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)
        if hasattr(timestep.observation, "action_mask"):
            legal_action_mask = timestep.observation.action_mask.astype(float)
        else:
            legal_action_mask = jnp.ones((self._num_actions,), dtype=float)
        if self._observation_attribute:
            agent_view = timestep.observation._asdict()[self._observation_attribute].astype(
                jnp.float32
            )
        else:
            agent_view = timestep.observation
        obs = Observation(agent_view, legal_action_mask, state.step_count)
        timestep_extras = timestep.extras
        if not timestep_extras:
            timestep_extras = {}
        timestep = timestep.replace(
            observation=obs,
            extras=timestep_extras,
        )
        return state, timestep

    #@cached_property # added line hbeyer; --> stoix other code needs this to be a function, instead of a property
    def observation_spec(self) -> Spec:
        if self._observation_attribute:
            agent_view_spec = Array(
                shape=self._env.observation_spec.__dict__[self._observation_attribute].shape, # fixed bug when using jumanji==1.1.0 instead of 1.0.0; changed observation_spec() --> observation_spec hbeyer
                dtype=float,
            )
        else:
            agent_view_spec = self._env.observation_spec # fixed bug when using jumanji==1.1.0 instead of 1.0.0; changed observation_spec() --> observation_spec hbeyer
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=agent_view_spec,
            action_mask=Array(shape=(self._num_actions,), dtype=float),
            step_count=Array(shape=(), dtype=int),
        )

    # added function hbeyer
    def action_spec(self) -> Spec:
        """ This has to be readded, as this property is overwritten by the Wrapper for Multidiscrete action spaces
            Also has to be readded to be called as an function, instead of the @cached_property in jumanji 1.1.0 (Stoix is for jumanji 1.0.0)
        """
        return self._env.action_spec


