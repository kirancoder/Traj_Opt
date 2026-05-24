# boundary_conditions.py

import numpy as np

def boundary_conditions(vars, N):
    x = vars[:N+1]
    y = vars[N+1:2*(N+1)]
    u = vars[2*(N+1):3*(N+1)]
    v = vars[3*(N+1):4*(N+1)]
    theta = vars[4*(N+1):5*(N+1)]
    m = vars[5*(N+1):6*(N+1)]

    bc = np.zeros(6)
    bc[0] = x[0] - 5  # x(0) = 5
    bc[1] = y[0] - 2  # y(0) = 2
    bc[2] = u[0] - 2  # u(0) = 2
    bc[3] = v[0] - 4  # v(0) = 4
    bc[4] = theta[0] - np.pi/40  # theta(0) = pi/40
    bc[5] = m[0] - 10000  # m(0) = 10000

    return np.concatenate((bc, [x[-1], y[-1], u[-1], v[-1], theta[-1]]))  # final conditions x, y, u, v, theta = 0
