import os
import numpy as np
import copy
import ast

from pydrake.all import PiecewisePolynomial

from qsim.examples.setup_simulations import run_quasistatic_sim
from qsim.simulator import ForwardDynamicsMode
from qsim.model_paths import models_dir

from qsim.parser import (
    QuasistaticParser,
    GradientMode,
    QuasistaticSystemBackend,
)

q_model_path = os.path.join(models_dir, "q_sys", "box_pushing.yml")

# # %% sim setup
# h = 0.01
# T = int(round(5.0 / h))  # num of time steps to simulate forward.
# duration = T * h
# print(duration)

# hand_name = "hand"
# object_name = "box"

# # trajectory and initial conditions.
# nq_a = 2
# qa_knots = np.zeros((2, nq_a))
# # x, y, z, dy1, dy2
# qa_knots[0] = [0.0, -0.2]
# qa_knots[1] = [0.1, 1.0]
# q_robot_traj = PiecewisePolynomial.FirstOrderHold([0, T * h], qa_knots.T)

# q_a_traj_dict_str = {hand_name: q_robot_traj}

# q_u0 = np.array([0, 0.7, 0])



# %% sim setup
h = 0.01

hand_name = "hand"
object_name = "box"

# --- Load xbar and ubar from results file ---
xbar = []
ubar = []
import json
results_file = os.path.join(os.path.dirname(__file__), "rddp2_box_push_results.json")
with open(results_file, "r") as f:
    results = json.load(f)
    xbar = np.array(results["xbar"])
    ubar = np.array(results["ubar"])
# Convert to numpy arrays
xbar = np.array(xbar)
ubar = np.array(ubar)

# Add 1000 repeated elements to ubar using last two elements of final xbar
final_hand_position = xbar[-1][-2:]  # Last two elements of final state
repeated_controls = np.tile(final_hand_position, (1000, 1))
ubar = np.vstack([ubar, repeated_controls])

# --- Set initial state and trajectory from loaded data ---
q0_dict_str = {object_name: np.array(xbar[0][:3]), hand_name: np.array(xbar[0][3:])}
T = ubar.shape[0]
h = 0.01
# Create time vector for trajectory
traj_times = np.arange(T) * h
q_robot_traj = PiecewisePolynomial.FirstOrderHold(traj_times, ubar.T)
q_a_traj_dict_str = {hand_name: q_robot_traj}

q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(
            is_quasi_dynamic=True,
            h=h,
            gravity=[0, 0, 0],
            gradient_mode=GradientMode.kAB,
            forward_mode=ForwardDynamicsMode.kSocpMp,
        )

loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
    q_parser=q_parser,
    h=h,
    backend=QuasistaticSystemBackend.CPP,
    q_a_traj_dict_str=q_a_traj_dict_str,
    q0_dict_str=q0_dict_str,
    is_visualizing=True,
    real_time_rate=1.0)


# # %% Calculate gradients.

# q_sim = q_parser.make_simulator_py()
# q_sim_cpp = q_parser.make_simulator_cpp()

# # %% look into the plant.
# plant = q_sim_cpp.get_plant()
# for model in q_sim_cpp.get_all_models():
#     print(
#         model,
#         plant.GetModelInstanceName(model),
#         q_sim_cpp.get_velocity_indices()[model],
#     )

# name_to_model_dict = q_sim_cpp.get_model_instance_name_to_index_map()
# idx_a = name_to_model_dict[hand_name]
# idx_u = name_to_model_dict[object_name]
# q_dict = {
#     idx_u: np.array([0, 0.7, 0]),
#     idx_a: np.array([-0.047203905846515,  -0.054111500180208]),
# }

# qa_cmd_dict = {idx_a: np.array([-0.047203905846515, -2.129185169602240])}

# # Params
# sim_params = copy.deepcopy(q_sim.sim_params)
# sim_params.forward_mode = ForwardDynamicsMode.kSocpMp
# sim_params.gradient_mode = GradientMode.kAB
# sim_params.h = h

# # CPP analytic gradients
# q_sim_cpp.update_mbp_positions(q_dict)
# tau_ext_dict = q_sim_cpp.calc_tau_ext([])
# q_sim_cpp.step(
#     q_a_cmd_dict=qa_cmd_dict, tau_ext_dict=tau_ext_dict, sim_params=sim_params
# )
# dfdu_active_cpp = q_sim_cpp.get_Dq_nextDqa_cmd()
# dfdq_active_cpp = q_sim_cpp.get_Dq_nextDq()

# pos = q_sim_cpp.get_mbp_positions_as_vec()
# # print(pos)

# # %% with printing gradients.

# output_file = "gradient_log.txt"

# # Reset initial state from initial conditions
# q_sim_cpp.update_mbp_positions(q_dict)

# # Sim params setup
# sim_params = copy.deepcopy(q_sim.sim_params)
# sim_params.forward_mode = ForwardDynamicsMode.kQpMp
# sim_params.gradient_mode = GradientMode.kAB
# sim_params.h = h

# with open(output_file, "w") as f:
#     f.write("Gradient Log per Timestep\n")
#     f.write("==========================\n")

#     for t in range(T):
#         t_curr = t * h
#         t_next = (t + 1) * h

#         # Command for this step from trajectory
#         qa_cmd = q_robot_traj.value(t_next).squeeze()
#         qa_cmd_dict = {idx_a: qa_cmd}

#         # External torques
#         tau_ext_dict = q_sim_cpp.calc_tau_ext([])

#         # Advance C++ sim
#         q_sim_cpp.step(q_a_cmd_dict=qa_cmd_dict, tau_ext_dict=tau_ext_dict, sim_params=sim_params)

#         # Get gradients
#         dfdu = q_sim_cpp.get_Dq_nextDqa_cmd()
#         dfdq = q_sim_cpp.get_Dq_nextDq()

#         # Get updated state from q_sim
#         pos_vec = q_sim_cpp.get_mbp_positions_as_vec()  # full [box(3), hand(2)]
        
#         print(qa_cmd)
#         print(pos_vec)

#         # # Log everything
#         # f.write(f"Time Step {t} ({t_curr:.2f}s → {t_next:.2f}s)\n")
#         # f.write("Current State [box_x, box_y, box_theta, hand_x, hand_y]:\n")
#         # f.write(np.array2string(pos_vec, formatter={'float_kind': lambda x: f"{x: .4f}"}))
#         # f.write("\n\n")

#         # f.write("df/dq:\n")
#         # f.write(np.array2string(dfdq, formatter={'float_kind': lambda x: f"{x: .4f}"}))
#         # f.write("\n\n")

#         # f.write("df/du:\n")
#         # f.write(np.array2string(dfdu, formatter={'float_kind': lambda x: f"{x: .4f}"}))
#         # f.write("\n")
#         # f.write("-" * 60 + "\n")

#         # Update state from current simulator for next step
#         q_next = q_sim_cpp.get_mbp_positions_as_vec()
#         q_dict[idx_u] = q_next[:3]
#         q_dict[idx_a] = q_next[3:]
#         q_sim_cpp.update_mbp_positions(q_dict)
