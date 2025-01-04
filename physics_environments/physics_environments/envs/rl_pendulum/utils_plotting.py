from typing import Tuple

import chex
import numpy as np

""" utils_data.py

      This file contains some utility methods for the data processing in viewer.py.
"""

def get_pxr_pyr_data(rod_lengths : chex.Array,
                     thetas      : chex.Array,
                     x_c_data    : chex.Array) -> Tuple[chex.Array]:
    """ Returns the cartesian coordinates of the rod tips

        It is not recommended to use this function for calculating agent feature inputs.
          - get_rod2cart_distance_features provides better information for the agent without x_c.
          - get_pxr_pyr_data provides redundant information after get_rod2cart_distance_features is used.
        Instead this function may only serve to get the cartesian rod positions for plotting and visualization applications.

        This function is only used in viewer.py at the time.
    """
    pxr_summands = -rod_lengths*np.sin(thetas) # element-wise product
    pyr_summands =  rod_lengths*np.cos(thetas)
    pxr_data = np.cumsum(pxr_summands) + x_c_data
    pyr_data = np.cumsum(pyr_summands)
    return pxr_data, pyr_data
