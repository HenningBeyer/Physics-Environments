from typing import Tuple

import chex
import jax.numpy as jnp

""" utils.py

      This file contains feature engineering methods for env._state_to_observation at the time.
      It is used by env.py only

"""

def get_base_features(bump_x : float,
                      s_x : chex.Numeric,
                      v_x : chex.Numeric,
                      a_x : chex.Numeric
                      ) -> chex.Array:
    """ Get a set common base features.
        Returns x position, velocity, and acceleration. Also returns x distance d_corner to the environment boundry.
    """
    d_corner = bump_x - jnp.abs(s_x)
    return jnp.array([s_x, v_x, a_x, d_corner])


def get_theta_features(thetas   : chex.Array,
                       dthetas  : chex.Array,
                       ddthetas : chex.Array
                       ) -> chex.Array:

    sin_thetas   =           jnp.sin(thetas)
    cos_thetas   =           jnp.cos(thetas)
    dsin_thetas  =   dthetas*jnp.cos(thetas)
    dcos_thetas  =  -dthetas*jnp.sin(thetas)
    ddsin_thetas =  ddthetas*jnp.cos(thetas) - dthetas**2*jnp.sin(thetas)
    ddcos_thetas = -ddthetas*jnp.sin(thetas) - dthetas**2*jnp.cos(thetas)
    return jnp.array([thetas, dthetas, ddthetas, sin_thetas, dsin_thetas, ddsin_thetas, cos_thetas, dcos_thetas, ddcos_thetas])



def get_rod2cart_distance_features(rod_lengths : chex.Array,
                                   thetas      : chex.Array,
                                   dthetas     : chex.Array,
                                   ddthetas    : chex.Array) -> Tuple[chex.Array]:
    """ Rod-to-cart distances.
        Rod 1: r2cy_1 = cos(θ_1)*l_1
        ...
        Rod 5: r2cy_3 = cos(θ_1)*l_1 + cos(θ_2)*l_2 + cos(θ_3)*l_3 + cos(θ_4)*l_4 + cos(θ_5)*l_5

        Note: a derivative of sin_thetas is mostly included with these features
    """

    # base features
    r2cx_summands = -rod_lengths*jnp.sin(thetas) # element-wise product
    r2cy_summands =  rod_lengths*jnp.cos(thetas)
    out_r2cx = jnp.cumsum(r2cx_summands)
    out_r2cy = jnp.cumsum(r2cy_summands)
    out_r2cd = jnp.sqrt(out_r2cx**2 + out_r2cy**2)

    # 1st derivative w.r.t t
    d_r2cx_summands = -rod_lengths*dthetas*jnp.cos(thetas)
    d_r2cy_summands = -rod_lengths*dthetas*jnp.sin(thetas)
    d_out_r2cx = jnp.cumsum(d_r2cx_summands)
    d_out_r2cy = jnp.cumsum(d_r2cy_summands)
    d_out_r2cd = (d_out_r2cx + d_out_r2cy)/(out_r2cd + 1e-8)

    # 2nd derivative w.r.t t
    dd_r2cx_summands = -rod_lengths*ddthetas*jnp.cos(thetas) + rod_lengths*dthetas**2*jnp.sin(thetas)
    dd_r2cy_summands = -rod_lengths*ddthetas*jnp.sin(thetas) + -rod_lengths*dthetas**2*jnp.cos(thetas)
    dd_out_r2cx = jnp.cumsum(dd_r2cx_summands)
    dd_out_r2cy = jnp.cumsum(dd_r2cy_summands)
    dd_out_r2cd = (dd_out_r2cx + dd_out_r2cy)/(out_r2cd + 1e-8) + \
                    (d_out_r2cx + d_out_r2cy)  *d_out_r2cd/(-0.5*out_r2cd**3 + 1e-8)

    return jnp.array([out_r2cx, d_out_r2cx, dd_out_r2cx, out_r2cy, d_out_r2cy, dd_out_r2cy, out_r2cd, d_out_r2cd, dd_out_r2cd])

def get_pxr_pyr_data(rod_lengths : chex.Array,
                     thetas      : chex.Array,
                     x_c_data    : chex.Array) -> Tuple[chex.Array]:
    """ Returns the cartesian coordinates of the rod tips

        It is NOT recommended to use this function for calculating agent feature inputs.
          - get_rod2cart_distance_features provides better information for the agent without x_c.
          - get_pxr_pyr_data provides redundant information after get_rod2cart_distance_features is used.
        Instead this function may only serve to get the cartesian rod positions for plotting and visualization applications.
    """
    pxr_summands = -rod_lengths*jnp.sin(thetas) # element-wise product
    pyr_summands =  rod_lengths*jnp.cos(thetas)
    pxr_data = jnp.cumsum(pxr_summands) + x_c_data
    pyr_data = jnp.cumsum(pyr_summands)
    return pxr_data, pyr_data