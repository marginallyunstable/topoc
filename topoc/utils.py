"""Utility Functions"""

from typing import Callable, Tuple, Any, Optional
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import functools

def linearize(fun: Callable) -> Callable:
    """
    Returns a Jacobian function with respect to x and u inputs,
    or just x if only one input. Handles normal functions and functools.partial.

    Returns:
    - For one argument: Callable[[Array], Array]
        Returns the Jacobian ∂f/∂x as an array.
    - For two arguments: Callable[[Array, Array], Tuple[Array, Array]]
        Returns the tuple of Jacobians (∂f/∂x, ∂f/∂u).
    Why wrapped?:
        It ensures that the function you pass to jax.jacrev always takes exactly the
        arguments you want to differentiate with respect to, regardless of whether the
        original function is a plain function or a functools.partial.
    """

    # Count the number of arguments the function expects after partial application
    def getnumargs(f):
        if isinstance(f, functools.partial):
            total = f.func.__code__.co_argcount
            num_fixed = len(f.args) + len(f.keywords) if f.keywords else len(f.args)
            return total - num_fixed
        return f.__code__.co_argcount

    nargs = getnumargs(fun)

    if nargs == 1:
        def wrapped(x):
            return fun(x)
        return jax.jacrev(wrapped, argnums=0)
    elif nargs == 2:
        def wrapped(x, u):
            return fun(x, u)
        return jax.jacrev(wrapped, argnums=(0, 1))
    else:
        raise ValueError("Function must take one or two arguments (x) or (x, u)")

def quadratize(fun: Callable) -> Callable:
    """
    Returns a Hessian function with respect to x and u inputs,
    or just x if only one input. Handles normal functions and functools.partial.

    Returns:
    - For one argument: Callable[[Array], Array]
        Returns the Hessian ∂²f/∂x² as an array.
    - For two arguments: Callable[[Array, Array], Tuple[Tuple[Array, Array], Tuple[Array, Array]]]
        Returns the block Hessian:
            (
                ( ∂²f/∂x², ∂²f/∂x∂u ),
                ( ∂²f/∂u∂x, ∂²f/∂u² )
            )
    """
    import functools

    def getnumargs(f):
        if isinstance(f, functools.partial):
            total = f.func.__code__.co_argcount
            num_fixed = len(f.args) + len(f.keywords) if f.keywords else len(f.args)
            return total - num_fixed
        return f.__code__.co_argcount

    nargs = getnumargs(fun)

    if nargs == 1:
        def wrapped(x):
            return fun(x)
        return jax.hessian(wrapped, argnums=0)
    elif nargs == 2:
        def wrapped(x, u):
            return fun(x, u)
        return jax.jacfwd(jax.jacrev(wrapped, argnums=(0, 1)), argnums=(0, 1))
    else:
        raise ValueError("Function must take one or two arguments (x) or (x, u)")
    

def quadratic_running_cost(x: Array, u: Array, params: Optional[Any] = None) -> Array:
    """
    Quadratic running cost: 0.5 * (x^T Q x + u^T R u)
    params should be a dict with 'Q' and 'R' arrays.
    """
    Q = params["Q"] if params and "Q" in params else jnp.eye(x.shape[0])
    R = params["R"] if params and "R" in params else jnp.eye(u.shape[0])
    cost = 0.5 * (x.T @ Q @ x + u.T @ R @ u)
    return cost

def quadratic_terminal_cost(x: Array, params: Optional[Any] = None) -> Array:
    """
    Quadratic terminal cost: 0.5 * (x^T P x)
    params should be a dict with 'P' array.
    """
    P = params["P"] if params and "P" in params else jnp.eye(x.shape[0])
    cost = 0.5 * (x.T @ P @ x)
    return cost