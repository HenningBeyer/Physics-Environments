from physics_environments.envs.rl_pendulum.cart_pendulum_math.math import CartPendulumMath
from physics_environments.envs.rl_pendulum.cart_pendulum_math.param_manager import CartPendulumParameterManager
from physics_environments.envs.rl_pendulum.cart_pendulum_math.sim_mixin import CartPendulumSimulationMixin
from physics_environments.envs.rl_pendulum.cart_pendulum_math.viz_mixin import CartPendulumVisualizationMixin

from physics_environments.envs.rl_pendulum.cart_pendulum_math.types import CartPendulumPhysicsConstants, CartPendulumPhysicsEnvironmentParams


class CartPendulumPhysicsEnvironment(CartPendulumParameterManager,         # Parent class that is responsible to store all potential parameters for the Pendulum environment and its mixin extensions.
                                     CartPendulumMath,                     # Parent class of Cart_Pendulum_Environment, which needs to be initialized to provide self.math
                                     CartPendulumSimulationMixin,          # Mixin class; uses self.math in its methods; provides simulation functions
                                     CartPendulumVisualizationMixin        # Mixin class that just adds methods; uses self.math in its methods; provides animation, plotting, and math summarization
                                    ):
    """ This is the full pendulum class with Cart_Pendulum_Parameter_Manager, Cart_Pendulum_Math and its optional mixin classes

        Input:
            - simply provide CartPendulumPhysicsEnvironmentParams()
    """
    def __init__(self,
                 params          : CartPendulumPhysicsEnvironmentParams = CartPendulumPhysicsEnvironmentParams(),
                 initialize_math : bool                                 = True,
                 ):
        """ Initializes Cart_Pendulum_Parameter_Manager and Cart_Pendulum_Math (parent classes)
            Mixin classes are not initialized; they have by definition no __init__().
        """
        CartPendulumParameterManager.__init__(self, params = params)
        CartPendulumMath.__init__(self, params=params['constants'], initialize_math = initialize_math)
        # All the other mixins provide all the functionalities of Cart_Pendulum_Environment by using self.math
        # Mixin class function get also their own parameters, but only when calling them by their methods

        # Saving the math core environment
        if params['save_after_init'] is True:
            self.save_class_obj_after_init(save_filename=params['save_filename'])

