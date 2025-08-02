import jax
import jax.numpy as jnp
from functools import partial
import time
from topoc.utils import *

# Example state and input
x = jnp.array([1.0, 2.0])
u = jnp.array([0.5, -1.0])

# Example Q and R matrices (can be diagonal or full)
P = jnp.array([[1.0, 0.0], [0.0, 1.0]])
Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
R = jnp.array([[0.0, 0.0], [0.0, 0.0]])

params = {"P": P}
terminalcost = partial(quadratic_terminal_cost, params=params)
params = {"Q": Q, "R": R}
runningcost = partial(quadratic_running_cost, params=params)

# Warm-up (important for JIT)
_ = runningcost(x, u)
_ = terminalcost(x)

# Timing the cost computation
start = time.time()
rcost = runningcost(x, u)
end = time.time()
print(f"Running cost computation time: {end - start:.8f} seconds")

start = time.time()
tcost = terminalcost(x)
end = time.time()
print(f"Terminal cost computation time: {end - start:.8f} seconds")

print("State x:", x)
print("Input u:", u)
print("Quadratic running cost (with baked-in Q, R):", rcost)
print("Quadratic terminal cost (with baked-in P):", tcost)

# --- Linearization (Jacobian) ---
linearized_runningcost = linearize(runningcost)
linearized_terminalcost = linearize(terminalcost)

jac_x, jac_u = linearized_runningcost(x, u)
print("Jacobian of running cost w.r.t x:", jac_x)
print("Jacobian of running cost w.r.t u:", jac_u)

jac_term_x = linearized_terminalcost(x)
print("Jacobian of terminal cost w.r.t x:", jac_term_x)

# --- Quadratization (Hessian) ---
quadratized_runningcost = quadratize(runningcost)
quadratized_terminalcost = quadratize(terminalcost)

# Hessian blocks of runningcost w.r.t. x and u
(hess_xx, hess_xu), (hess_ux, hess_uu) = quadratized_runningcost(x, u)
print("Hessian of running cost w.r.t x,x:", hess_xx)
print("Hessian of running cost w.r.t x,u:", hess_xu)
print("Hessian of running cost w.r.t u,x:", hess_ux)
print("Hessian of running cost w.r.t u,u:", hess_uu)

# Hessian of terminalcost w.r.t. x
hess_term_xx = quadratized_terminalcost(x)
print("Hessian of terminal cost w.r.t x,x:", hess_term_xx)
