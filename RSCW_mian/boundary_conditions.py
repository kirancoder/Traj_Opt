# boundary_conditions.py

import numpy as np
from config import x_0, y_0, u_0, v_0, theta_0, m_0

def boundary_conditions(vars, N):
    x = vars[:N+1]
    y = vars[N+1:2*(N+1)]
    u = vars[2*(N+1):3*(N+1)]
    v = vars[3*(N+1):4*(N+1)]
    theta = vars[4*(N+1):5*(N+1)]
    m = vars[5*(N+1):6*(N+1)]

    bc = np.zeros(6)
    bc[0] = x[0] - x_0  # x(0) = x_0
    bc[1] = y[0] - y_0  # y(0) = y_0
    bc[2] = u[0] - u_0  # u(0) = u_0
    bc[3] = v[0] - v_0  # v(0) = v_0
    bc[4] = theta[0] - theta_0  # theta(0) = theta_0
    bc[5] = m[0] - m_0  # m(0) = m_0

    return np.concatenate((bc, [x[-1], y[-1], u[-1], v[-1], theta[-1]]))  # final conditions x, y, u, v, theta = 0
