"""
Simple test script for BoxPushSimulator dynamics function.
"""

import jax.numpy as jnp
from models.qsim_box_push import BoxPushSimulator

# Create the simulator
dt = 0.01
box = BoxPushSimulator(dt)

# Test inputs
x = jnp.array([0.0, 0.7, 0.0, 0.0, 0.0])  # State: [box_x, box_y, box_z, hand_x, hand_y]
u = jnp.array([0.0, 0.0])                  # Control: [hand_dx, hand_dy]

print("Testing BoxPushSimulator dynamics function")
print("=" * 50)
print(f"Input state (x): {x}")
print(f"Input control (u): {u}")
print()

# Test the dynamics function
try:
    next_state = box.dynamics(x, u)
    print(f"Output next state: {next_state}")
    print(f"State shape: {next_state.shape}")
    print(f"State dtype: {next_state.dtype}")
    print()
    
    # Show the change
    state_change = next_state - x
    print(f"State change (next - current): {state_change}")
    
    print("\n✓ Dynamics function test completed successfully!")
    
except Exception as e:
    print(f"❌ Error testing dynamics function: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Testing BoxPushSimulator graddynamics function")
print("=" * 50)

# Create batch of states and controls for graddynamics
x_batch = jnp.array([[0.0, 0.7, 0.0, 0.0, 0.0],   # State 1: [box_x, box_y, box_z, hand_x, hand_y]
                     [0.0, 0.7, 0.0, 0.0, 0.0]])  # State 2: slightly different position
u_batch = jnp.array([[0.0, 0.0],                   # Control 1: [hand_dx, hand_dy]
                     [0.0, 0.0]])                  # Control 2: small movement

print(f"Input state batch (x_batch): {x_batch}")
print(f"Input control batch (u_batch): {u_batch}")
print(f"Batch size: {x_batch.shape[0]}")
print()

# Test the graddynamics function
try:
    A_batch, B_batch = box.graddynamics(x_batch, u_batch, std_u=0.01, mcsamples=20)
    print(f"A_batch (state Jacobian): {A_batch}")
    print(f"A_batch shape: {A_batch.shape}")
    print(f"A_batch dtype: {A_batch.dtype}")
    print()
    
    print(f"B_batch (control Jacobian): {B_batch}")
    print(f"B_batch shape: {B_batch.shape}")
    print(f"B_batch dtype: {B_batch.dtype}")
    print()
    
    print("✓ Graddynamics function test completed successfully!")
    
except Exception as e:
    print(f"❌ Error testing graddynamics function: {e}")
    import traceback
    traceback.print_exc()
