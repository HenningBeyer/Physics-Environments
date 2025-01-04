from typing import Any, Dict

import dill
import numpy as np

from physics_environments.envs.rl_pendulum.cart_pendulum_math.types import CartPendulumPhysicsEnvironmentParams

class CartPendulumParameterManager():
    """ Class which unpacks parameters and checks them for correct definition.
        It also pre-defines every possible attribute in advance for clarity and has no effect on other classes.
        This is the first class instantiated when instantiating Cart_Pendulum_Environment class.

        This class is also very useful for validity checks requested by the UI.
    """
    def __init__(self,
                 params : Dict[str, Any]):

        # Parameter unpacking:
        self.n                      = params.constants.n
        self.n_range                = range(1, self.n+1)   # np.arange(1, self.n+1) # fix because of omegaconf ListConfig ...
        self.n_plus_1_range         = range(1, self.n+1+1) # np.arange(1, self.n+1+1)
        self.env_type               = params.env_type  # one of ['inverted pendulum', 'compound pendulum']

        # Cart_Pendulum_Math attributes
        self.constants             = {} # string-symbol-value mappings for params.constants
        self.variables             = {} # t and \ddot{x_c}
        self.math                  = {} # definition of all string-symbol-function mappings
        self.substitution_mappings = {} # other useful mappings for substitutions

        # Cart_Pendulum_Feature_Engineering_Mixin
        self.state_functions       = {} # symbols-function mappings for calculating state-features for feature engineering

        # Cart_Pendulum_Simulation_Mixin
        self.state                 = {} # simulation state of a single step; set in self.step(), reset in self.reset()

        self.__parameter_sanity_check(params)

    def __parameter_sanity_check(self,
                                 params : CartPendulumPhysicsEnvironmentParams
                                 ) -> None:
        constants      = params.constants
        # Checking type definitions
        assert (self.env_type in ('inverted pendulum', 'compound pendulum')), ("env_type must be either 'inverted pendulum' or 'compound pendulum'")

        # Checking constant definitions
        list_defined_constants = ['l', 'r', 'm', 'mu', 'I']
        for c_ in list_defined_constants:
            assert (len(constants[c_])  == self.n), (f"{constants[c_]} needs to have n={self.n} entries for {c_}")

        if 'r' not in constants: # Parameter correction of r without warning, if r not defined
            constants.r = constants.l # r will be heavily used for modeling the 'inverted pendulum', so it is always needed.
        if self.env_type == 'inverted pendulum':
            assert (np.prod(constants.r == constants.l)), (f"Every masspoint distance r={constants.r} should be exactly set to l={constants.l}, when using type 'inverted pendulum'")
        if self.env_type == 'compound pendulum':
            assert (np.prod(np.array(constants.r) <= np.array(constants.l))), ('every r needs to fullfill r <= l')

    def save_class_obj_after_init(self, save_filename : str) -> None:
        """ Saves the Cart_Pendulum_Environment class (this class initializes Cart_Pendulum_Environment class by class and then saves the class bundle), if self.save_after_init is set to True.
            Saves the hard to calculate math of Cart_Pendulum_Environment, avoiding long loading times of 5 or more minutes.

            Input:
                save_filename:
                    - follows the structure: f"env{self.constants.n}n_{self.env_type}_obj.pkl"
                    - example: 'env_n4_inverted_pendulum_obj.pkl'   for n=4, and env_type='inverted_pendulum'
                    - has to have a .pkl extension.
        """
        dill.settings['recurse'] = True         # Allows to bypass some errors
        with open(save_filename, 'wb') as file: # elf.save_filename needs a .pkl extension
            dill.dump(self, file)               # dump the self object

    @staticmethod
    def load_env_obj_from_file(filename : str):
        """ Reads a Cart_Pendulum_Environment class object from an automatically saved .pkl file.
            Bypasses long initialization times of that class.

            One can access this static method like CartPendulumParameterManager.load_env_obj_from_file(file_name)
                - This does not require to initialize CartPendulumParameterManager!

            Input:
                filename: the string filename of the file to load
                    - Example: 'env_n4_inverted_pendulum_12fgs_obj.pkl'
        """
        with open(filename, 'rb') as file:
            pendulum_env = dill.load(file)
        return pendulum_env
