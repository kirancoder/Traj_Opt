# config
import jax.numpy as jnp

# Define the problem parameters
N = 10  # Reduced number of collocation points for debugging
a = 100
c1 = 10
c2 = 2
g = 9.81
m = 1.0  # Mass


# Initial conditions
x_0 = 2.0
y_0 = 10.0
u_0 = 0.0
v_0 = 0.0
theta_0 = jnp.pi / 10 

# Final conditions
x_f = 0.0
y_f = 0.0
u_f = 0.0
v_f = 0.0
theta_f = 0.0