from jumanji.registration import register

# Definition of the module interface:
## (Importing all interfacable classes here)
from physics_environments.envs.rl_pendulum.env import RLCartPendulumEnvironment
from physics_environments.envs.rl_pendulum.cart_pendulum_math.env import CartPendulumPhysicsEnvironment

from physics_environments.envs.rl_pendulum.cart_pendulum_math.types import CartPendulumPhysicsConstants, CartPendulumPhysicsEnvironmentParams
from physics_environments.envs.rl_pendulum.types import RLCartPendulumEnvironmentParams

from physics_environments.version import __version__

# Defining the from rl_pendulum import * imports:
# __all__ = ['RLCartPendulumEnvironment', 'CartPendulumPhysicsEnvironment']

# Registering all environments made with jumanji:
register(id="cart_n_pendulum-v1", entry_point="physics_environments.envs.rl_pendulum:RLCartPendulumEnvironment")