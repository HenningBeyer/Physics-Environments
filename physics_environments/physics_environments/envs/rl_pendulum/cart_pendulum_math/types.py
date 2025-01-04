from typing import List, Literal
from dataclasses import dataclass, field
from physics_environments.types import ParamsBaseMixin

@dataclass
class CartPendulumPhysicsConstants(ParamsBaseMixin):
    """ Copy-Paste Template:
            n = 2
            constants = CartPendulumPhysicsConstants(n    = n,
                                                     g    = 9.81,
                                                     l    = n*[0.300],
                                                     r    = n*[0.200],
                                                     m    = n*[0.800],
                                                     I    = n*[0.011],
                                                     mu   = n*[0.015])

        Parameter Definition:
            n    : int          = 2             --> number of rods; core parameter; IMPORTANT: One has to specify the last 5 params [l,r,m,\mu,I] with lists of length n!
            g    : float        = 9.81          --> g constant
            l    : List[float]  = n*[0.300])    --> rod lengths
            r    : List[float]  = n*[0.200])    --> center-of-mass distances per rod; only used if 'env_type': 'compound pendulum', else set to values of r to l for 'inverted pendulum'
            m    : List[float]  = n*[0.800])    --> rod link masses; m would be chosen depending on the material and rod length
            I    : List[float]  = n*[0.011])    --> rod inertias; Formulas: (I_i = 1/3*m_i*l_i**2) or (I_i = 1/3*m_i*r_i**2) (values match the experimental data of the source notebook)
            mu   : List[float]  = n*[0.015])    --> rod link frictions



    """

    n    : int          = 1
    g    : float        = 9.81
    l    : List[float]  = field(default_factory=lambda n_=n: n_*[0.300])
    r    : List[float]  = field(default_factory=lambda n_=n: n_*[0.200])
    m    : List[float]  = field(default_factory=lambda n_=n: n_*[0.800])
    I    : List[float]  = field(default_factory=lambda n_=n: n_*[0.011])
    mu   : List[float]  = field(default_factory=lambda n_=n: n_*[0.015])


@dataclass
class CartPendulumPhysicsEnvironmentParams(ParamsBaseMixin):
    """ Copy-Paste Template:
            env_type             = 'compound pendulum'
            cart_pendulum_params = CartPendulumPhysicsEnvironmentParams(constants       = constants,
                                                                        env_type        = env_type,
                                                                        save_after_init = True,
                                                                        save_filename   = f"env_{n}n_{env_type}_obj.pkl")

        Parameter Definition:

            constants         : CartPendulumPhysicsConstants \
                                    = field(default_factory=CartPendulumPhysicsConstants)

            env_type          : Literal['compound pendulum', 'inverted pendulum'] \
                                     = 'compound pendulum'                          --> type of the pendulum; only 'compound pendulum' considered; legacy: 'inverted pendulum' (can be fully modelled using 'compound pendulum')
            save_after_init   : bool = True                                         --> Wheter to automatically save the symbolic derivations as .pkl objects; these can take very long to calculate
            save_filename     : str  = None                                         --> use a custom filename or a default like 'env_n4_inverted_pendulum_obj.pkl'

            def __post_init__(self):
                if self.save_filename is None:
                    self.save_filename = f"env_{self.constants.n}n_{self.env_type}_obj.pkl"
    """

    constants         : CartPendulumPhysicsConstants \
                             = field(default_factory=CartPendulumPhysicsConstants)

    env_type          : str  = 'compound pendulum'                          # type of the pendulum; only 'compound pendulum' considered; legacy: 'inverted pendulum' (can be fully modelled using 'compound pendulum')
                                                                            # Literal['compound pendulum', 'inverted pendulum'] --> removed this typing as it causes problems with omegaconfig
    save_after_init   : bool = False                                        # Wheter to automatically save the symbolic derivations as .pkl objects; these can take very long to calculate
    save_filename     : str  = None                                         # use a custom filename or a default like 'env_n4_inverted_pendulum_obj.pkl'

    def __post_init__(self):
        if self.save_filename is None:
            self.save_filename = f"env{self.constants.n}n_{self.env_type}_obj.pkl"
