import numpy as np
from cyipopt import minimize_ipopt
from jax import jit, grad, hessian, jacfwd, jacrev
import jax.numpy as jnp
from config import N
from Untils import objective, dynamics, boundary_conditions
from post_process import plot_trajectory

# Initial guess of states
x0 = np.zeros(N+1)
y0 = np.zeros(N+1)
u0 = np.zeros(N+1)
v0 = np.zeros(N+1)

# Initial guess of control inputs
theta0 = np.zeros(N+1)
U1_0 = np.zeros(N+1)
U2_0 = np.zeros(N+1)

# Initial guess of final time
tf0 = 10

# Convert initial guess to JAX arrays
initial_guess = jnp.concatenate([
    jnp.array(u0), jnp.array(v0), jnp.array(x0), jnp.array(y0), 
    jnp.array(theta0), jnp.array(U1_0), jnp.array(U2_0), jnp.array([tf0])
])

# Bounds for variables
bounds = [(-np.inf, np.inf)]*(N+1) + [(-np.inf, np.inf)]*(N+1) \
         + [(0, np.inf)]*(N+1) + [(0, np.inf)]*(N+1) \
         + [(-np.inf, np.inf)]*(N+1) + [(0.05, 1)]*(N+1) \
         + [(-1, 1)]*(N+1) + [(0, np.inf)]

# ---- Compute objective, constraints, and their derivatives ----

# JIT compile the objective function, dynamics, and boundary conditions
obj_jit = jit(objective)

con_dyn_jit = jit(dynamics)

con_bc_jit = jit(boundary_conditions)

# Define the gradient (Jacobian) and Hessian of the objective function
def objective_gradient(vars):
    return grad(obj_jit)(vars)

objective_grad_jit = jit(objective_gradient)

def all_constraints(vars):
    return jnp.concatenate([
        dynamics(vars),
        boundary_conditions(vars)
    ])

con_all_jit = jit(all_constraints)

constraints_jacobian = jit(jacfwd(all_constraints))

result = minimize_ipopt(
    fun=obj_jit,
    x0=np.array(initial_guess),
    jac=objective_grad_jit,
    bounds=bounds,
    constraints={
        "type": "eq",
        "fun": con_all_jit,
        "jac": constraints_jacobian,
    },
    options={
        "disp": 5,
        "max_iter": 500
    }
)

# Extract results
optimal_vars = result.x
optimal_t_f = optimal_vars[-1]

# Post-process the results
plot_trajectory(optimal_vars, N, title="Optimal Trajectory")