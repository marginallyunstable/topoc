import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from topoc.utils import quadratic_running_cost, quadratic_terminal_cost, plot_compare_block_results
from topoc.base import TOProblemDefinition, TOAlgorithm, TOSolve
from models.block import block_on_ground, block_on_ground_with_friction
from topoc.types import ModelParams, AlgorithmName
import os
import numpy as np



# Define model parameters (example values)
state_dim = 2
input_dim = 1
horizon_len = 200
dt = 0.01

modelparams = ModelParams(
    state_dim=state_dim,
    input_dim=input_dim,
    horizon_len=horizon_len,
    dt=dt
)

# Define initial and goal states
x0 = jnp.array([0.0, 0.0])
covx0 = 1e-6 * jnp.eye(state_dim)
xg = jnp.array([4.0, 0.0])
# Define initial input (control)
u0 = jnp.array([0.0])

# Define cost matrices
P = 1000000*jnp.eye(state_dim)
Q = 1*jnp.eye(state_dim)
R = 50*jnp.eye(input_dim)

params_dynamics = {"m": 1.0, "dt": dt}
params_terminal = {"P": P}
params_running = {"Q": Q, "R": R}

# Define cost functions using partial
dynamics = partial(block_on_ground_with_friction, params=params_dynamics)
terminalcost = partial(quadratic_terminal_cost, xg=xg, params=params_terminal)
runningcost = partial(quadratic_running_cost, xg=xg, params=params_running)

# region DDP

toprob_ddp = TOProblemDefinition(
    runningcost=runningcost,
    terminalcost=terminalcost,
    dynamics=dynamics,
    starting_state=x0,
    starting_input=u0,
    goal_state=xg,
    modelparams=modelparams
)

algorithm_ddp = TOAlgorithm(AlgorithmName.DDP, use_second_order_info=True, max_iters=50)

tosolve_ddp = TOSolve(toprob_ddp, algorithm_ddp)
xbar_ddp, ubar_ddp, Vstore_ddp = tosolve_ddp.result.xbar, tosolve_ddp.result.ubar, tosolve_ddp.result.Vstore

# endregion DDP


# region SCSDDP

toprob_scsddp = TOProblemDefinition(
    runningcost=runningcost,
    terminalcost=terminalcost,
    dynamics=dynamics,
    starting_state=x0,
    starting_input=u0,
    goal_state=xg,
    modelparams=modelparams
)

# Define SCSDDP algorithm parameters (example values)
algorithm_scsddp = TOAlgorithm(
    AlgorithmName.SCSDDP,
    use_second_order_info=False, # NOTE
    sigma_x=1e-6,
    sigma_u=10,
    alpha=0.1,
    alpha_red=2.0,
    sigma_red=2.0,
    targetalpha=1e-6,
    targetsigma=1e-6,
    max_iters=50,
    spg_method='gh_ws',
    spg_params={"order": 6},
)

# Example usage: create and solve the problem with SCSDDP
tosolve_scsddp = TOSolve(toprob_scsddp, algorithm_scsddp)
xbar_scsddp, ubar_scsddp, Vstore_scsddp = tosolve_scsddp.result.xbar, tosolve_scsddp.result.ubar, tosolve_scsddp.result.Vstore


# endregion SCSDDP


# region CSDDP

toprob_csddp = TOProblemDefinition(
    runningcost=runningcost,
    terminalcost=terminalcost,
    dynamics=dynamics,
    starting_state=x0,
    starting_input=u0,
    goal_state=xg,
    modelparams=modelparams
)

# Define CSDDP algorithm parameters (example values)
algorithm_csddp = TOAlgorithm(
    AlgorithmName.CSDDP,
    use_second_order_info=False,
    sigma=10,
    alpha=0.1,
    alpha_red=2.0,
    sigma_red=2.0,
    targetalpha=1e-6,
    targetsigma=1e-6,
    max_iters=50,
    spg_method='gh_ws',
    spg_params={"order": 6},
)

tosolve_csddp = TOSolve(toprob_csddp, algorithm_csddp)
xbar_csddp, ubar_csddp, Vstore_csddp = tosolve_csddp.result.xbar, tosolve_csddp.result.ubar, tosolve_csddp.result.Vstore


# endregion CSDDP

# region PDDP

toprob_pddp = TOProblemDefinition(
    runningcost=runningcost,
    terminalcost=terminalcost,
    dynamics=dynamics,
    starting_state=x0,
    starting_state_cov=covx0,
    starting_input=u0,
    goal_state=xg,
    modelparams=modelparams
)

# Define PDDP algorithm parameters (example values)
algorithm_pddp = TOAlgorithm(
    AlgorithmName.PDDP,
    use_second_order_info=False,
    spg_method='gh_ws',
    spg_params={"order": 6},
    eta=0.01,
    lam=201,
    zeta=1,
    zeta_factor=2,
    zeta_min=1e-2,
    sigma_u=10,
    max_iters=50
)


# Example usage: create and solve the problem with PDDP
tosolve_pddp = TOSolve(toprob_pddp, algorithm_pddp)
xbar_pddp, ubar_pddp, Vstore_pddp = tosolve_pddp.result.xbar, tosolve_pddp.result.ubar, tosolve_pddp.result.Vstore

# endregion PDDP

# recreate algorithms list and call the beautified plotting function
algorithms = [
    ("DDP", xbar_ddp, ubar_ddp, Vstore_ddp),
    ("SCS-DDP", xbar_scsddp, ubar_scsddp, Vstore_scsddp),
    ("CS-DDP", xbar_csddp, ubar_csddp, Vstore_csddp),
    ("PDDP", xbar_pddp, ubar_pddp, Vstore_pddp),
]

plot_compare_block_results(algorithms, x0, xg, modelparams, friction_thresh=7.848)



# region: Save all algorithm results in one categorized file

os.makedirs("results", exist_ok=True)
def _sanitized_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")

def save_all_results(algorithms, dt, base_dir="results", filename="all_algorithms_solution.npz"):
    """
    Save all algorithm results into a single .npz file.
    The file contains keys of the form:
      <sanitized_name>_xbar, <sanitized_name>_ubar, <sanitized_name>_Vstore
    and metadata keys:
      algorithm_names (original names), sanitized_names, dt
    """
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, filename)

    data_dict = {}
    alg_names = []
    san_names = []

    for name, xbar, ubar, Vstore in algorithms:
        sname = _sanitized_name(name)
        alg_names.append(name)
        san_names.append(sname)

        data_dict[f"{sname}_xbar"] = np.array(xbar)
        data_dict[f"{sname}_ubar"] = np.array(ubar)
        data_dict[f"{sname}_Vstore"] = np.array(Vstore)

    data_dict["algorithm_names"] = np.array(alg_names, dtype=object)
    data_dict["sanitized_names"] = np.array(san_names, dtype=object)
    data_dict["dt"] = np.array(dt)

    np.savez(filepath, **data_dict)
    return filepath


algorithms_to_save = [
    ("DDP", xbar_ddp, ubar_ddp, Vstore_ddp),
    ("SCS-DDP", xbar_scsddp, ubar_scsddp, Vstore_scsddp),
    ("CS-DDP", xbar_csddp, ubar_csddp, Vstore_csddp),
    ("PDDP", xbar_pddp, ubar_pddp, Vstore_pddp),
]

saved_path = save_all_results(algorithms_to_save, dt, x0=x0, xg=xg)
# Update save_all_results to accept x0 and xg and save them directly
def save_all_results(algorithms, dt, x0=None, xg=None, base_dir="results", filename="block_with_friction_solution.npz"):
    """
    Save all algorithm results into a single .npz file.
    The file contains keys of the form:
      <sanitized_name>_xbar, <sanitized_name>_ubar, <sanitized_name>_Vstore
    and metadata keys:
      algorithm_names (original names), sanitized_names, dt, x0, xg
    """
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, filename)

    data_dict = {}
    alg_names = []
    san_names = []

    for name, xbar, ubar, Vstore in algorithms:
        sname = _sanitized_name(name)
        alg_names.append(name)
        san_names.append(sname)

        data_dict[f"{sname}_xbar"] = np.array(xbar)
        data_dict[f"{sname}_ubar"] = np.array(ubar)
        data_dict[f"{sname}_Vstore"] = np.array(Vstore)

    data_dict["algorithm_names"] = np.array(alg_names, dtype=object)
    data_dict["sanitized_names"] = np.array(san_names, dtype=object)
    data_dict["dt"] = np.array(dt)

    if x0 is not None:
        data_dict["x0"] = np.array(x0)
    if xg is not None:
        data_dict["xg"] = np.array(xg)

    np.savez(filepath, **data_dict)
    return filepath

# Save all algorithm results in one categorized file and include start/goal
saved_path = save_all_results(algorithms_to_save, dt, x0=x0, xg=xg)
print(f"Saved combined results to: {saved_path}")

# endregion Save results to a single .npz file