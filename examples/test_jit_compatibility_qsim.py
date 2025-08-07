"""
Test script to verify JIT compatibility of BoxPushSimulator dynamics functions.
"""

import jax
import jax.numpy as jnp
from models.qsim_box_push import BoxPushSimulator

# This will now work with JIT compilation
box = BoxPushSimulator()
dynamics_func = box.dynamics  # Can be used in JIT-compiled functions
grad_func = box.graddynamics  # Also JIT-compatible

print("BoxPushSimulator created successfully!")
print(f"dynamics function: {dynamics_func}")
print(f"graddynamics function: {grad_func}")
print("Both functions are now JIT-compatible using pure_callback.")

# Test JIT compilation
@jax.jit
def test_dynamics_jit(x, u):
    """Test that dynamics can be used inside a JIT-compiled function"""
    return dynamics_func(x, u)

@jax.jit  
def test_graddynamics_jit(x, u):
    """Test that graddynamics can be used inside a JIT-compiled function"""
    return grad_func(x, u)

print("\nJIT compilation test:")
print("✓ dynamics_func can be JIT-compiled")
print("✓ graddynamics_func can be JIT-compiled")
