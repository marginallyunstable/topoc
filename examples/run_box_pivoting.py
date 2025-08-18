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

q_model_path = os.path.join("/workspace/models", "q_sys", "box_pivoting.yml")

# %% sim setup
h = 0.1
T = int(round(5.0 / h))  # num of time steps to simulate forward.
duration = T * h
print(duration)

hand_name = "hand"
object_name = "box"

# trajectory and initial conditions.
nq_a = 2
qa_knots = np.zeros((2, nq_a))
# x, y, z, dy1, dy2
qa_knots[0] = [-0.7, 0.5]
qa_knots[1] = [-0.7, 0.5]
# qa_knots[1] = [1.0, 1.2]
q_robot_traj = PiecewisePolynomial.FirstOrderHold([0, T * h], qa_knots.T)


q_a_traj_dict_str = {"hand": q_robot_traj}

q_u0 = np.array([0, 0.7, 0])
q0_dict_str = {"box": q_u0, "hand": qa_knots[0]}

q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(
            is_quasi_dynamic=True,
            h=h,
            # gravity=[0, 0, 0],
            contact_detection_tolerance=0.1,
            gradient_mode=GradientMode.kAB,
            forward_mode=ForwardDynamicsMode.kQpMp,
        )
loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
    q_parser=q_parser,
    h=h,
    backend=QuasistaticSystemBackend.CPP,
    q_a_traj_dict_str=q_a_traj_dict_str,
    q0_dict_str=q0_dict_str,
    is_visualizing=True,
    real_time_rate=1.0)
