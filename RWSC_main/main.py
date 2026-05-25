# main.py
import os
import numpy as np
import pandas as pd
# from scipy.optimize import minimize
from cyipopt import minimize_ipopt
from jax import jit, grad, jacfwd, jacrev
from dynamics import dynamics
from boundary_conditions import boundary_conditions

# Define the problem parameters
N = 20  # Number of collocation points
g = 9.81
c1 = 44000
c2 = 311 * g
c3 = 0.0698
gamma = 1.5e-2

# Initial guesses for state variables
x0 = np.linspace(5, 0, N+1)
y0 = np.linspace(2, 0, N+1)
u0 = np.linspace(2, 0, N+1)
v0 = np.linspace(4, 0, N+1)
theta0 = np.linspace(np.pi/40, 0, N+1)
m0 = np.linspace(10000, 8000, N+1)
U1_0 = np.linspace(0, 1, N+1)
U2_0 = np.linspace(-1, 1, N+1)
tf0 = 10.0  # Initial guess of final time

# Concatenate all initial guesses
initial_guess = np.concatenate([x0, y0, u0, v0, theta0, m0, U1_0, U2_0, [tf0]])

# Define bounds for the variables
bounds = [(-np.inf, np.inf)]*(N+1) + [(0, np.inf)]*(N+1) + [(-np.inf, np.inf)]*(N+1) + [(-np.inf, np.inf)]*(N+1) + [(-np.pi/2, np.pi/2)]*(N+1) + [(0, np.inf)]*(N+1) + [(0, 1)]*(N+1) + [(-1, 1)]*(N+1) + [(0, np.inf)]

# Define the new objective function
# main.py

@jit
def objective(vars):
    U1 = vars[6*(N+1):7*(N+1)]
    U2 = vars[7*(N+1):8*(N+1)]
    t_f = vars[-1]

    # Time step size
    dt = t_f / N

    # Compute integrand at each collocation point
    integrand = gamma * (c1 * U1 / c2)

    # Hermite-Simpson integration
    cost = 0
    for i in range(N):
        # Midpoint control and integrand
        U1_mid = 0.5 * (U1[i] + U1[i+1])
        integrand_mid = gamma * (c1 * U1_mid / c2)

        # Apply Hermite-Simpson rule
        cost += (dt / 6) * (integrand[i] + 4 * integrand_mid + integrand[i+1])

    # Compute U1_dot using finite differences
    U1_dot = np.zeros(N+1)
    U1_dot[0] = (-3 * U1[0] + 4 * U1[1] - U1[2]) / (2 * dt)
    for i in range(1, N):
        U1_dot[i] = (U1[i+1] - U1[i-1]) / (2 * dt)
    U1_dot[-1] = (3 * U1[-1] - 4 * U1[-2] + U1[-3]) / (2 * dt)

    # Compute U2_dot using finite differences
    U2_dot = np.zeros(N+1)
    U2_dot[0] = (-3 * U2[0] + 4 * U2[1] - U2[2]) / (2 * dt)
    for i in range(1, N):
        U2_dot[i] = (U2[i+1] - U2[i-1]) / (2 * dt)
    U2_dot[-1] = (3 * U2[-1] - 4 * U2[-2] + U2[-3]) / (2 * dt)

    # Compute the squared norm of U1_dot and U2_dot
    U1_dot_norm_sq = np.sum(U1_dot**2)
    U2_dot_norm_sq = np.sum(U2_dot**2)

    # Add regularization terms to the cost function
    regularization_weight = 1.0  # Adjust this weight as needed
    cost += regularization_weight * (U1_dot_norm_sq + U2_dot_norm_sq)

    return cost



# Combine dynamics and boundary conditions into a single constraint function

def constraints(vars):
    dyn_res = dynamics(vars, N, c1, c2, c3, g)
    bond_res = boundary_conditions(vars, N)
    return np.concatenate((dyn_res, bond_res))

# Define constraints for the solver
cons = {
    'type': 'eq',
    'fun': constraints,
}

# Solve the problem
result = minimize_ipopt(objective, 
                  initial_guess, 
                  method='trust-constr', 
                  constraints=cons,
                  bounds=bounds,  
                  options={
                            'disp': True,
                            'maxiter': 10000,
                            'gtol': 1e-10  # Set a higher gradient tolerance
                  }
                  )

# Extract the solution
w_opt = result.x
x_opt = w_opt[:N+1]
y_opt = w_opt[N+1:2*(N+1)]
u_opt = w_opt[2*(N+1):3*(N+1)]
v_opt = w_opt[3*(N+1):4*(N+1)]
theta_opt = w_opt[4*(N+1):5*(N+1)]
m_opt = w_opt[5*(N+1):6*(N+1)]
U1_opt = w_opt[6*(N+1):7*(N+1)]
U2_opt = w_opt[7*(N+1):8*(N+1)]
t_f_opt = w_opt[-1]

print("Optimal final time:", t_f_opt)

# Generate time vector
time = np.linspace(0, t_f_opt, N+1)

# Create a DataFrame with the results
df = pd.DataFrame({
    'Time': time,
    'x': x_opt,
    'y': y_opt,
    'u': u_opt,
    'v': v_opt,
    'theta': theta_opt,
    'm': m_opt,
    'U1': U1_opt,
    'U2': U2_opt
})

print("Current working directory:", os.getcwd())
# Save the DataFrame to a CSV file
try:
    df.to_csv('RWSC_reag.csv', index=False)
    print("CSV file created successfully.")
except Exception as e:
    print(f"An error occurred while creating the CSV file: {e}")

