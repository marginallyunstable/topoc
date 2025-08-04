"""Block on ground models"""

from typing import Any, Optional
from jax import Array
import jax.numpy as jnp

# region: Models

def block_on_ground(x: Array, u: Array, params: Optional[Any] = None) -> Array:
    """
    Block on ground model
    """
    m = params["m"] if params and "m" in params else 1.0
    dt = params["dt"] if params and "dt" in params else 0.01
    xnext = x + dt * jnp.array([x[1], u[0] / m])
    return xnext

def block_on_ground_with_friction(x: Array, u: Array, params: Optional[Any] = None) -> Array:
    """
    Block on ground model with friction (JAX-friendly).
    Implements sticking, sliding, and breakaway friction.
    """
    m = params["m"] if params and "m" in params else 1.0
    g = params["g"] if params and "g" in params else 9.81
    mu = params["mu"] if params and "mu" in params else 0.8
    dt = params["dt"] if params and "dt" in params else 0.01

    fric_max = mu * m * g

    v = x[1]
    I = m
    v_free = v + dt * (u[0]) / I
    v_thresh = dt * fric_max / I

    # Friction law (sticking, sliding, breakaway)
    # sticking: abs(v_free) <= v_thresh -> friction = -I*v_free/dt
    # sliding: v_free > v_thresh -> friction = -fric_max
    # sliding: v_free < -v_thresh -> friction = fric_max

    friction = jnp.where(
        jnp.abs(v_free) <= v_thresh,
        -I * v_free / dt,
        jnp.where(
            v_free > v_thresh,
            -fric_max,
            fric_max
        )
    )

    x2_next = v + (dt / I) * (u[0] + friction)
    x1_next = x[0] + dt * x2_next

    xnew = jnp.array([x1_next, x2_next])
    return xnew

# endregion: Models