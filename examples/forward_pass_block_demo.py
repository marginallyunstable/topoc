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

# --- Without JIT: forward_pass ---
start = time.time()
(new_Xs, new_Us), total_cost = forward_pass(Xs, Us, Ks, ks, toproblem)
end = time.time()
# print("Simulated states:\n", new_Xs)
# print("Simulated controls:\n", new_Us)
# print("Total cost:", total_cost)
print(f"forward_pass (no JIT) time: {end - start:.8f} seconds")

# --- With JIT: forward_pass (toproblem as static argument) ---
forward_pass_jit = jax.jit(forward_pass, static_argnums=4)
_ = forward_pass_jit(Xs, Us, Ks, ks, toproblem)  # Warm-up
start = time.time()
(new_Xs_jit, new_Us_jit), total_cost_jit = forward_pass_jit(Xs, Us, Ks, ks, toproblem)
end = time.time()
# print("Simulated states [JIT]:\n", new_Xs_jit)
# print("Simulated controls [JIT]:\n", new_Us_jit)
# print("Total cost [JIT]:", total_cost_jit)
print(f"forward_pass (with JIT) time: {end - start:.8f} seconds")

# --- Without JIT: forward_pass ---
start = time.time()
(new_Xs, new_Us), total_cost = forward_pass(Xs, Us, Ks, ks, toproblem)
end = time.time()
# print("Simulated states:\n", new_Xs)
# print("Simulated controls:\n", new_Us)
# print("Total cost:", total_cost)
print(f"second forward_pass (no JIT) time: {end - start:.8f} seconds")

# --- Without JIT: traj_batch_derivatives ---
start = time.time()
traj_derivs = traj_batch_derivatives(Xs, Us, toproblem)
end = time.time()
print("traj_batch_derivatives (no JIT) time: {:.8f} seconds".format(end - start))

# --- With JIT: traj_batch_derivatives (toproblem as static argument) ---
traj_batch_derivatives_jit = jax.jit(traj_batch_derivatives, static_argnums=2)
_ = traj_batch_derivatives_jit(Xs, Us, toproblem)  # Warm-up
start = time.time()
traj_derivs_jit = traj_batch_derivatives_jit(Xs, Us, toproblem)
end = time.time()
print("traj_batch_derivatives (with JIT) time: {:.8f} seconds".format(end - start))

# print("\ntraj_batch_derivatives (with JIT) values:")
# print("fxs:\n", traj_derivs_jit.fxs)
# print("fus:\n", traj_derivs_jit.fus)
# print("fxxs:\n", traj_derivs_jit.fxxs)
# print("fxus:\n", traj_derivs_jit.fxus)
# print("fuxs:\n", traj_derivs_jit.fuxs)
# print("fuus:\n", traj_derivs_jit.fuus)
# print("lxs:\n", traj_derivs_jit.lxs)
# print("lus:\n", traj_derivs_jit.lus)
# print("lxxs:\n", traj_derivs_jit.lxxs)
# print("lxus:\n", traj_derivs_jit.lxus)
# print("luxs:\n", traj_derivs_jit.luxs)
# print("luus:\n", traj_derivs_jit.luus)
# print("lfx:\n", traj_derivs_jit.lfx)
# print("lfxx:\n", traj_derivs_jit.lfxx)

# --- Without JIT: backward_pass ---
start = time.time()
dV, success, K_seq, k_seq, Vx_seq, Vxx_seq = backward_pass(traj_derivs)
end = time.time()
print("\nbackward_pass (no JIT) time: {:.8f} seconds".format(end - start))
# print("dV:", dV)
# print("success:", success)
# print("K_seq:\n", K_seq)
# print("k_seq:\n", k_seq)
# print("Vx_seq:\n", Vx_seq)
# print("Vxx_seq:\n", Vxx_seq)

# --- With JIT: backward_pass ---
backward_pass_jit = jax.jit(backward_pass)
_ = backward_pass_jit(traj_derivs)  # Warm-up
start = time.time()
dV_jit, success_jit, K_seq_jit, k_seq_jit, Vx_seq_jit, Vxx_seq_jit = backward_pass_jit(traj_derivs)
end = time.time()
print("\nbackward_pass (with JIT) time: {:.8f} seconds".format(end - start))
# print("dV [JIT]:", dV_jit)
# print("success [JIT]:", success_jit)
# print("K_seq [JIT]:\n", K_seq_jit)
# print("k_seq [JIT]:\n", k_seq_jit)
# print("Vx_seq [JIT]:\n", Vx_seq_jit)
# print("Vxx_seq [JIT]:\n", Vxx_seq_jit)

# --- Example call to forward_iteration ---
class DummyAlgorithm:
    class Params:
        gamma = 0.01
        beta = 0.5
    params = Params()

Vprev = total_cost_jit
dV = dV_jit 
dummy_algo = DummyAlgorithm()

start = time.time()
Xs_new, Us_new, V_new, eps_used, done_flag = forward_iteration(
    new_Xs_jit, new_Us_jit, K_seq_jit, k_seq_jit, Vprev, dV, toproblem, dummy_algo
)
end = time.time()
print("\nforward_iteration results:")
# print("Xs_new:\n", Xs_new)
# print("Us_new:\n", Us_new)
# print("V_new:", V_new)
# print("eps_used:", eps_used)
# print("done_flag:", done_flag)
print("forward_iteration time: {:.8f} seconds".format(end - start))

# --- JIT version of forward_iteration ---
forward_iteration_jit = jax.jit(forward_iteration, static_argnums=(6, 7, 8))
_ = forward_iteration_jit(
    new_Xs_jit, new_Us_jit, K_seq_jit, k_seq_jit, Vprev, dV, toproblem, dummy_algo
)  # Warm-up
start = time.time()
Xs_new_jit, Us_new_jit, V_new_jit, eps_used_jit, done_flag_jit = forward_iteration_jit(
    new_Xs_jit, new_Us_jit, K_seq_jit, k_seq_jit, Vprev, dV, toproblem, dummy_algo
)
Xs_new_jit.block_until_ready()  # Ensure all computation is finished
end = time.time()
print("\nforward_iteration [JIT] results:")
# print("Xs_new [JIT]:\n", Xs_new_jit)
# print("Us_new [JIT]:\n", Us_new_jit)
# print("V_new [JIT]:", V_new_jit)
# print("eps_used [JIT]:", eps_used_jit)
# print("done_flag [JIT]:", done_flag_jit)
print("forward_iteration [JIT] time: {:.8f} seconds".format(end - start))
