import os
import numpy as np
import jax.numpy as jnp
import jax
from jax.experimental import io_callback
import copy
from qsim.simulator import ForwardDynamicsMode
from qsim.model_paths import models_dir
from qsim.parser import (
    QuasistaticParser,
    GradientMode,
)

class PlanarHandSimulator:
    def __init__(self, dt=0.01):
        q_model_path = os.path.join(models_dir, "q_sys", "planar_hand_ball.yml")
        self.robot_l_name = "arm_left"
        self.robot_r_name = "arm_right"
        self.object_name = "sphere"
        self.q_parser = QuasistaticParser(q_model_path)
        self.q_parser.set_sim_params(
            is_quasi_dynamic=True,
            h=dt,
            gravity=[0, 0, 0],
            gradient_mode=GradientMode.kAB,
            forward_mode=ForwardDynamicsMode.kSocpMp,
        )
        self.q_sim = self.q_parser.make_simulator_py()
        self.q_sim_cpp = self.q_parser.make_simulator_cpp()
        self.sim_params = copy.deepcopy(self.q_sim.sim_params)
        self.q_sim_batch = self.q_parser.make_batch_simulator()
        name_to_model_dict = self.q_sim_cpp.get_model_instance_name_to_index_map()
        self.idx_a_l = name_to_model_dict[self.robot_l_name]
        self.idx_a_r = name_to_model_dict[self.robot_r_name]
        self.idx_u = name_to_model_dict[self.object_name]
        self.seed = 42

    def dynamics(self, x, u):
        def _dynamics_impl(x_np, u_np):
            # first three elements of x are the box position, next two are the hand position
            q_dict = {self.idx_u: x_np[:3], self.idx_a_l: x_np[3:5], self.idx_a_r: x_np[5:]}
            qa_cmd_dict = {self.idx_a_l: u_np[:2], self.idx_a_r: u_np[2:]}
            self.q_sim_cpp.update_mbp_positions(q_dict)
            tau_ext_dict = self.q_sim_cpp.calc_tau_ext([])
            self.q_sim_cpp.step(
                q_a_cmd_dict=qa_cmd_dict, tau_ext_dict=tau_ext_dict, sim_params=self.sim_params
            )
            next_pos = self.q_sim_cpp.get_mbp_positions_as_vec()
            return next_pos
        
        # Use io_callback to make it JIT-compatible
        result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
        return io_callback(_dynamics_impl, result_shape, x, u)

    def graddynamics(self, x, u, std_u = 0.01, mcsamples = 50):
        def _graddynamics_impl(x_np, u_np, std_u_val, mcsamples_val):
            dim_u = u_np.shape[-1]
            std_u_arr = np.full(dim_u, std_u_val)

            A_list, B_list, c_list = self.q_sim_batch.calc_bundled_ABc_trj(
                x_np, u_np, std_u_arr, self.sim_params, mcsamples_val, self.seed
            )
            
            # Convert lists of matrices to 3D numpy arrays
            # A_list: list of N matrices, each (n_x, n_x) -> (N, n_x, n_x)
            # B_list: list of N matrices, each (n_x, n_u) -> (N, n_x, n_u)
            A_batch = np.stack(A_list, axis=0).astype(np.float32)
            B_batch = np.stack(B_list, axis=0).astype(np.float32)
            
            return A_batch, B_batch
        
        # Use io_callback to make it JIT-compatible
        # A_batch and B_batch shapes need to be specified
        # A_batch: (N, n_x, n_x), B_batch: (N, n_x, n_u)
        N = x.shape[0]
        n_x = x.shape[1]
        n_u = u.shape[1]
        A_shape = jax.ShapeDtypeStruct((N, n_x, n_x), jnp.float32)
        B_shape = jax.ShapeDtypeStruct((N, n_x, n_u), jnp.float32)
        
        # io_callback returns a tuple when result_shape is a tuple
        return io_callback(
            _graddynamics_impl, 
            (A_shape, B_shape),
            x, u, std_u, mcsamples
        )