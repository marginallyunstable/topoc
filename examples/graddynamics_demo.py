from models.block.block import block_on_ground_with_friction
from topoc.utils import linearize, quadratize
from functools import partial
import jax
import jax.numpy as jnp
import time

params = {"m": 1.0, "g": 9.81, "mu": 0.8, "dt": 0.01}
dyn = partial(block_on_ground_with_friction, params=params)

# Example state and input
x = jnp.array([1.0, 2.0])
u = jnp.array([0.5])

# --- Linearize (Jacobian) ---
jac_dyn = linearize(dyn)

# Without JIT
start = time.time()
fx, fu = jac_dyn(x, u)
end = time.time()
print("Jacobian w.r.t x (fx):\n", fx)
print("Jacobian w.r.t u (fu):\n", fu)
print(f"Linearize (no JIT) time: {end - start:.8f} seconds")

# With JIT
jac_dyn_jit = jax.jit(jac_dyn)
_ = jac_dyn_jit(x, u)  # Warm-up
start = time.time()
fx_jit, fu_jit = jac_dyn_jit(x, u)
end = time.time()
print("Jacobian w.r.t x (fx) [JIT]:\n", fx_jit)
print("Jacobian w.r.t u (fu) [JIT]:\n", fu_jit)
print(f"Linearize (with JIT) time: {end - start:.8f} seconds")

# --- Quadratize (Hessian) ---
hess_dyn = quadratize(dyn)

# Without JIT
start = time.time()
(fxx, fxu), (fux, fuu) = hess_dyn(x, u)
end = time.time()
print("Hessian w.r.t x,x (fxx):\n", fxx)
print("Hessian w.r.t x,u (fxu):\n", fxu)
print("Hessian w.r.t u,x (fux):\n", fux)
print("Hessian w.r.t u,u (fuu):\n", fuu)
print(f"Quadratize (no JIT) time: {end - start:.8f} seconds")

# With JIT
hess_dyn_jit = jax.jit(hess_dyn)
_ = hess_dyn_jit(x, u)  # Warm-up
start = time.time()
(fxx_jit, fxu_jit), (fux_jit, fuu_jit) = hess_dyn_jit(x, u)
end = time.time()
print("Hessian w.r.t x,x (fxx) [JIT]:\n", fxx_jit)
print("Hessian w.r.t x,u (fxu) [JIT]:\n", fxu_jit)
print("Hessian w.r.t u,x (fux) [JIT]:\n", fux_jit)
print("Hessian w.r.t u,u (fuu) [JIT]:\n", fuu_jit)
print(f"Quadratize (with JIT) time: {end - start:.8f} seconds")
