import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from topoc.utils import gh_ws
from topoc.base import TOProblemDefinition, TOAlgorithm, TOSolve
from models.block import block_on_ground, block_on_ground_with_friction
from topoc.types import ModelParams, AlgorithmName


WI, XI = gh_ws(2,10)

print("WI:", WI)
print("XI:", XI)    