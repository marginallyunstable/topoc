"""Block on ground models"""

from typing import Any, Optional
from jax import Array
import jax.numpy as jnp

# region: Models

def pendulum(x: Array, u: Array, params: Optional[Any] = None) -> Array:
    """
    Pendulum model
    """
    m = params["m"] if params and "m" in params else 1.0
    l = params["l"] if params and "l" in params else 1.0
    g = params["g"] if params and "g" in params else 9.81
    dt = params["dt"] if params and "dt" in params else 0.01
    
    xnext = x + dt * jnp.array([
        x[1],
        3 * g * jnp.sin(x[0]) / (2 * l) + 3 * u[0] / (m * l ** 2)
    ])
    return xnext

def pendulum_with_friction(x: Array, u: Array, params: Optional[Any] = None) -> Array:
    """
    Pendulum model with friction (JAX-friendly).
    Implements sticking, sliding, and breakaway friction.
    """
    m = params["m"] if params and "m" in params else 1.0
    l = params["l"] if params and "l" in params else 1.0
    g = params["g"] if params and "g" in params else 9.81
    mu = params["mu"] if params and "mu" in params else 0.8
    dt = params["dt"] if params and "dt" in params else 0.01

    fric_max = 5

    I = 1/3 * m * l**2
    tau_g = 0.5* m * g * l * jnp.sin(x[0])
    v_free = x[1] + dt * (u[0] + tau_g) / I
    v = x[1]
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

    x2_next = v + (dt / I) * (u[0] + tau_g + friction)
    x1_next = x[0] + dt * x2_next

    xnew = jnp.array([x1_next, x2_next])
    return xnew

# endregion: Models