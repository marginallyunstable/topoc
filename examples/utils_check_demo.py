import jax
import jax.numpy as jnp
from functools import partial
from models.block.block import block_on_ground
from topoc.utils import backward_pass, forward_pass, forward_iteration, traj_batch_derivatives, linearize, quadratize
import time
import jax.profiler

# Define problem parameters
params = {"m": 1.0, "dt": 0.1}
dynamics = partial(block_on_ground, params=params)
runningcost = lambda x, u: jnp.sum(x**2) + jnp.sum(u**2)
terminalcost = lambda x: jnp.sum(x**2)

class DummyProblem:
    def __init__(self):
        self.dynamics = dynamics
        self.runningcost = runningcost
        self.terminalcost = terminalcost
        class ModelParams:
            dt = 0.1
        self.modelparams = ModelParams()
        # Use linearize and quadratize for derivatives
        self.graddynamics = linearize(self.dynamics)
        self.hessiandynamics = quadratize(self.dynamics)
        self.gradrunningcost = linearize(self.runningcost)
        self.hessianrunningcost = quadratize(self.runningcost)
        self.gradterminalcost = linearize(self.terminalcost)
        self.hessiantterminalcost = quadratize(self.terminalcost)

# Simulation horizon and dimensions
T = 150
n = 2
m = 1

# Initial state and zero controls
x0 = jnp.array([1.0, 0.0])
Xs = jnp.vstack([x0, jnp.zeros((T, n))])
Us = jnp.ones((T, m))
Ks = jnp.zeros((T, m, n))
ks = jnp.zeros((T, m))

toproblem = DummyProblem()

# --- Warm-up calls for JIT compilation ---
_ = forward_pass(Xs, Us, Ks, ks, toproblem)
_ = traj_batch_derivatives(Xs, Us, toproblem)

# --- Timed: forward_pass (JIT-decorated) ---
start = time.time()
(new_Xs, new_Us), total_cost = forward_pass(Xs, Us, Ks, ks, toproblem)
end = time.time()
print(f"Total cost: {total_cost}")
print(f"forward_pass time: {end - start:.8f} seconds")

# --- Timed: traj_batch_derivatives (JIT-decorated) ---
start = time.time()
traj_derivs = traj_batch_derivatives(new_Xs, new_Us, toproblem)
end = time.time()
print("traj_batch_derivatives time: {:.8f} seconds".format(end - start))

# --- Timed: backward_pass (JIT-decorated) ---
start = time.time()
dV, success, K_seq, k_seq, Vx_seq, Vxx_seq = backward_pass(traj_derivs)
end = time.time()
print("\nbackward_pass time: {:.8f} seconds".format(end - start))

# --- Example call to forward_iteration (JIT-decorated) ---
class DummyAlgorithm:
    class Params:
        gamma = 0.01
        beta = 0.5
    params = Params()

Vprev = total_cost
dV = dV
dummy_algo = DummyAlgorithm()

# Warm-up
_ = forward_iteration(new_Xs, new_Us, K_seq, k_seq, Vprev, dV, toproblem, dummy_algo)

start = time.time()
Xs_new, Us_new, V_new, eps_used, done_flag = forward_iteration(
    new_Xs, new_Us, K_seq, k_seq, Vprev, dV, toproblem, dummy_algo
)
Xs_new.block_until_ready()  # Ensure all computation is finished
end = time.time()
print("\nforward_iteration results:")
print("forward_iteration time: {:.8f} seconds".format(end - start))
print("V_new:", V_new)

# Optionally, print results for inspection
# print("Simulated states:\n", new_Xs)
# print("Simulated controls:\n", new_Us)
# print("Total cost:", total_cost)
# print("traj_batch_derivatives values:", traj_derivs)
# print("dV:", dV)
# print("success:", success)
# print("K_seq:\n", K_seq)
# print("k_seq:\n", k_seq)
# print("Vx_seq:\n", Vx_seq)
# print("Vxx_seq:\n", Vxx_seq)
# print("Xs_new:\n", Xs_new)
# print("Us_new:\n", Us_new)
# print("V_new:", V_new)
# print("eps_used:", eps_used)
# print("done_flag:", done_flag)