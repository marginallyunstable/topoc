"""Cartpole models"""

from typing import Any, Optional
from jax import Array
import jax.numpy as jnp
import jax

# region: Models

def cartpole(x, u, params):
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


def cartpole_with_friction(x, u, params):
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