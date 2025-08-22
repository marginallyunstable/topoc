"""Block on ground models"""

from typing import Any, Optional
from jax import Array
import jax.numpy as jnp
import jax

# region: Models

def cartpole(x: Array, u: Array, params: Optional[Any] = None) -> Array:
    """
    Pendulum model
    """
    mc = params["mc"] if params and "mc" in params else 1.0
    mp = params["mp"] if params and "mp" in params else 0.1
    l = params["l"] if params and "l" in params else 1.0
    g = params["g"] if params and "g" in params else 9.81
    dt = params["dt"] if params and "dt" in params else 0.01
    
    pos = x[0]
    vel = x[1]
    theta = x[2]
    theta_dot = x[3]
    F = u[0]

    temp = (F + mp * l * theta_dot**2 * jnp.sin(theta)) / (mc + mp)
    numerator = g * jnp.sin(theta) - jnp.cos(theta) * temp
    denominator = l * (4.0 / 3.0 - mp * jnp.cos(theta)**2 / (mc + mp))
    theta_dot_dot = numerator / denominator
    x_dot_dot = temp - mp * l * theta_dot_dot * jnp.cos(theta) / (mc + mp)

    f_continuous = jnp.array([vel, x_dot_dot, theta_dot, theta_dot_dot])
    xnext = x + f_continuous * dt
    return xnext

def cartpole__(x, u, params):

    mp = params['mp'] if params and "mp" in params else 1.0
    mc = params['mc'] if params and "mc" in params else 10.0
    pl = params['pl'] if params and "l" in params else 0.5
    g  = params['g'] if params and "g" in params else 9.81

    # x1, theta, dx, dtheta = x
    x1, dx, theta, dtheta = x
    s = jnp.sin(theta)
    c = jnp.cos(theta)

    C = jnp.array([[0, -mp * pl * dtheta * s],
                   [0, 0]])
    G = jnp.array([0, mp * g * pl * s])
    B = jnp.array([1, 0])
    H = jnp.array([[mc + mp, mp * pl * c],
                   [mp * pl * c, mp * pl ** 2]])


    Hinv = jnp.array([
        [1 / (-mp * c**2 + mc + mp),
         -c / (-mp * pl * c**2 + mc * pl + mp * pl)],
        [-c / (-mp * pl * c**2 + mc * pl + mp * pl),
         (mc + mp) / (-c**2 * mp**2 * pl**2 + mp**2 * pl**2 + mc * mp * pl**2)]
    ])

    qdd = Hinv @ (B * u - C @ jnp.array([dx, dtheta]) - G)
    y = jnp.array([dx, qdd[0], dtheta, qdd[1]])

    return y

def rk_f(x, u, dt, f):

    dx1 = f(x, u) * dt
    dx2 = f(x + 0.5 * dx1, u) * dt
    dx3 = f(x + 0.5 * dx2, u) * dt
    dx4 = f(x + dx3, u) * dt
    return x + (1.0 / 6.0) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)

def cartpole_with_friction(x: Array, u: Array, params: Optional[Any] = None) -> Array:
    """
    Block on ground model with friction (JAX-friendly).
    Implements sticking, sliding, and breakaway friction.
    """
    mc = params["mc"] if params and "mc" in params else 1.0
    mp = params["mp"] if params and "mp" in params else 0.1
    l = params["l"] if params and "l" in params else 1.0
    g = params["g"] if params and "g" in params else 9.81
    dt = params["dt"] if params and "dt" in params else 0.01

    fric_max = 0.9
    pos = x[0]
    vel = x[1]
    theta = x[2]
    theta_dot = x[3]
    F = u[0]

    temp = (F + mp * l * theta_dot**2 * jnp.sin(theta)) / (mc + mp)
    numerator = g * jnp.sin(theta) - jnp.cos(theta) * temp
    denominator = l * (4.0 / 3.0 - mp * jnp.cos(theta)**2 / (mc + mp))
    theta_dot_dot = numerator / denominator
    theta_dot_free = theta_dot + dt * theta_dot_dot
    theta_dot_thresh = dt * fric_max / (mp * l) / denominator

    # jax.debug.print("theta_dot_free: {}", theta_dot_free)
    # jax.debug.print("theta_dot_thresh: {}", theta_dot_thresh)

    # Friction law (sticking, sliding, breakaway)
    # sticking: abs(v_free) <= v_thresh -> friction = -I*v_free/dt
    # sliding: v_free > v_thresh -> friction = -fric_max
    # sliding: v_free < -v_thresh -> friction = fric_max

    theta_dot_dot_new = jnp.where(
        jnp.abs(theta_dot_free) <= theta_dot_thresh,
        0.0,
        jnp.where(
            theta_dot_free > theta_dot_thresh,
            (numerator - (fric_max/(mp*l)))/denominator,
            (numerator + (fric_max/(mp*l)))/denominator
        )
    )

    vel_dot_next = (F+mp*l*(theta_dot**2 * jnp.sin(theta) - theta_dot_dot_new*jnp.cos(theta)))/(mc+mp)
    
    vel_next = vel + dt*vel_dot_next
    theta_dot_next = theta_dot + dt*theta_dot_dot_new
    pos_next = pos + dt*vel_next
    theta_next = theta + dt*theta_dot_next

    xnew = jnp.array([pos_next, vel_next, theta_next, theta_dot_next])
    return xnew

def cartpole_f(x, u, params):
    mp = params['mp'] if params and "mp" in params else 0.1
    mc = params['mc'] if params and "mc" in params else 1.0
    pl = params['l'] if params and "l" in params else 0.5
    g = params['g'] if params and "g" in params else 9.81
    dt = params["dt"] if params and "dt" in params else 0.01
    
    # Extract state variables
    pos = x[0]
    vel = x[1]
    theta = x[2]
    theta_dot = x[3]
    F = u[0]
    
    # Trigonometric terms
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    
    # Mass matrix M(q)
    M = jnp.array([
        [mc + mp, mp * pl * c],
        [mp * pl * c, mp * pl**2]
    ])
    
    # Coriolis and gravity terms h(q, v)
    h = jnp.array([
        -mp * pl * s * theta_dot**2,
        mp * g * pl * s
    ])
    
    # External force vector (without friction)
    Q = jnp.array([F, 0.0])
    
    # Compute M inverse
    det_M = mp * pl**2 * (mc + mp * s**2)
    M_inv = jnp.array([
        [mp * pl**2, -mp * pl * c],
        [-mp * pl * c, mc + mp]
    ]) / det_M
    
    # Free velocity update (S1)
    acceleration_free = M_inv @ (Q - h)
    v_free = jnp.array([vel, theta_dot]) + dt * acceleration_free
    
    # Position update (UP)
    q_next = jnp.array([
        pos + dt * v_free[0],  # x position
        v_free[0],             # x velocity
        theta + dt * v_free[1], # theta angle
        v_free[1]              # theta velocity
    ])
    
    return q_next


def cartpole_f_with_friction(x, u, params):
    mp = params['mp'] if params and "mp" in params else 0.1
    mc = params['mc'] if params and "mc" in params else 1.0
    pl = params['l'] if params and "l" in params else 0.5
    g = params['g'] if params and "g" in params else 9.81
    dt = params["dt"] if params and "dt" in params else 0.01
    fricstatic_max = params["fricstatic_max"] if params and "fricstatic_max" in params else 0.2  # maximum static friction torque
    fricdynamic_max = params["fricdynamic_max"] if params and "fricdynamic_max" in params else 0.2  # maximum dynamic friction torque
    
    # Extract state variables
    pos = x[0]
    vel = x[1]
    theta = x[2]
    theta_dot = x[3]
    F = u[0]
    
    # Trigonometric terms
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    
    # Mass matrix M(q)
    M = jnp.array([
        [mc + mp, mp * pl * c],
        [mp * pl * c, mp * pl**2]
    ])
    
    # Coriolis and gravity terms h(q, v)
    h = jnp.array([
        -mp * pl * s * theta_dot**2,
        mp * g * pl * s
    ])
    
    # External force vector (without friction)
    Q_no_friction = jnp.array([F, 0.0])
    
    # Compute M inverse
    det_M = mp * pl**2 * (mc + mp * s**2)
    M_inv = jnp.array([
        [mp * pl**2, -mp * pl * c],
        [-mp * pl * c, mc + mp]
    ]) / det_M
    
    # Free velocity update (S1)
    acceleration_free = M_inv @ (Q_no_friction - h)
    v_free = jnp.array([vel, theta_dot]) + dt * acceleration_free
    
    # Extract free angular velocity ω*_{k+1}
    omega_free = v_free[1]
    
    # Angular impulse-velocity gain S(q_k) (S3)
    S_gain = M_inv[1, 1]  # e2^T M^{-1} e2
    
    # Stick condition test (T2)
    stick_condition = jnp.abs(omega_free) <= S_gain * fricstatic_max * dt

    # # Debug print omega_free and S_gain * fricstatic_max * dt
    # jax.debug.print("omega_free: {}", omega_free)
    # jax.debug.print("S_gain * fricstatic_max * dt: {}", S_gain * fricstatic_max * dt)
    
    # Stick impulse (T1)
    gamma_stick = -omega_free / S_gain
    
    # Slip impulse (K3)
    sign_omega_free = jnp.sign(omega_free)
    # Handle case where omega_free = 0 by using previous sign
    sign_omega_free = jnp.where(omega_free == 0.0, jnp.sign(theta_dot), sign_omega_free)
    gamma_slip = -fricdynamic_max * dt * sign_omega_free
    
    # Choose impulse based on condition
    gamma = jnp.where(stick_condition, gamma_stick, gamma_slip)
    
    # Apply impulse to get final velocity (S2)
    impulse_effect = M_inv @ jnp.array([0.0, gamma])
    v_final = v_free + impulse_effect
    
    # Position update (UP)
    q_next = jnp.array([
        pos + dt * v_final[0],  # x position
        v_final[0],             # x velocity
        theta + dt * v_final[1], # theta angle
        v_final[1]              # theta velocity
    ])
    
    return q_next

# endregion: Models