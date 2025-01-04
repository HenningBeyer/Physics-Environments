import abc

import chex
import jax.numpy as jnp
import jax.lax as lax

from physics_environments.envs.rl_pendulum.types import State


class DoneFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State, next_state: State, action: chex.Array) -> chex.Array:
        """Call method for computing the done signal given the current and next state,
        and the action taken.
        """

class TerminateDoneFn(DoneFn):
    """ Terminate the episode as soon as the environment is solved or the episode ended as intended. """
    def __init__(self, t_episode : float):
        self.t_episode = t_episode

    def __call__(self, state: State) -> chex.Array:
        """ End naturally if reaching the episode end. """
        return (state.t >= self.t_episode)

class TruncateDoneFn(DoneFn):
    """ Truncate the episode as soon as an invalid action is taken, a forbidden state was reached, or the goal could not be reached in time by an agent. """
    def __init__(self, bump_x: float):
        self.bump_x = bump_x

    def __call__(self, state: State) -> chex.Array:
        """ End if any cart bumps into the side """
        return (jnp.abs(state.s_x) > self.bump_x)