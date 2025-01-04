import abc

import chex
import jax.numpy as jnp

from physics_environments.envs.rl_pendulum.types import State

class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State) -> chex.Array:
        """Call method for computing the reward given current state and selected action."""


class DefaultRewardFn(RewardFn):
    """ """

    def __init__(
        self,
        bumping_corner_distance: float,
        reward_exponent        : float = 1 # higher values scale the reward such that rod andgles different to the standing position are rewarded significantly less; thus promotes high accuracy objectives
    ):
        self.bumping_corner_distance = bumping_corner_distance
        self.reward_exponent         = reward_exponent

    def __call__(self, state: State) -> chex.Array:
        angles       = state.thetas
        angles       = angles - (angles // (jnp.pi*2))*jnp.pi*2   # Convert n*[-2pi; +2pi] net angles --> to [0, 2pi], while accounting for n net angles (-inf; +inf)
        angles       = (angles + jnp.pi) % (2 * jnp.pi) - jnp.pi  # Convert [0, 2pi] --> [-pi, pi]
        rod_rewards  = jnp.cos(angles)                            # [-pi, pi] angles --> [-1, 1] rewards (standing rods have angle of 0, hanging rods hang with angle of pi)
        reward       = jnp.mean(rod_rewards)                      # ranges of [-1, 1]
        reward       = ((reward + 1)/2)**self.reward_exponent     # ranges of [0, 1] and can be squared, etc. such that higher precision is rewarded more

        #add_rewards1 = jnp.where(jnp.abs(state.s_x) > self.bumping_corner_distance, -15, 0)  # add big penalty if outside pendulum width; big penalty, as episode is directly terminated after that.

        reward       = reward #+ add_rewards1
        return reward