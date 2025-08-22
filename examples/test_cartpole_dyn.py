import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from topoc.utils import quadratic_running_cost, quadratic_terminal_cost, plot_cartpole_results
from topoc.base import TOProblemDefinition, TOAlgorithm, TOSolve
from models.cartpole import cartpole, cartpole_with_friction, cartpole_f, cartpole_f_with_friction  # Assumes this exists
from topoc.types import ModelParams, AlgorithmName
from jax import jit

# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    params = {
        'mp': 0.1,
        'mc': 1.0,
        'l': 0.5,
        'g': 9.81,
        'dt': 0.01,
        'fricstatic_max': 0.9,
        'fricdynamic_max': 0.8
    }
    
    # Test state and input
    x = jnp.array([0.0, 0.0, 0.1, 0.0])  # [pos, vel, theta, theta_dot]
    u = jnp.array([20.0])  # Force on cart
    
    # JIT compile the function
    jitted_func = jit(cartpole_f_with_friction)
    
    # Test the function
    result = jitted_func(x, u, params)
    print("Next state:", result)
    
    # Test with different conditions
    # High angular velocity (should slip)
    x_slip = jnp.array([0.0, 0.0, 0.1, 5.0])
    result_slip = jitted_func(x_slip, u, params)
    print("Slip case result:", result_slip)
    
    # Low angular velocity (should stick)
    x_stick = jnp.array([0.0, 0.0, 0.1, 0.01])
    result_stick = jitted_func(x_stick, u, params)
    print("Stick case result:", result_stick)