# dynamics.py

import numpy as np

def dynamics(vars, N, c1, c2, c3, g):
    x = vars[:N+1]
    y = vars[N+1:2*(N+1)]
    u = vars[2*(N+1):3*(N+1)]
    v = vars[3*(N+1):4*(N+1)]
    theta = vars[4*(N+1):5*(N+1)]
    m = vars[5*(N+1):6*(N+1)]
    U1 = vars[6*(N+1):7*(N+1)]
    U2 = vars[7*(N+1):8*(N+1)]
    t_f = vars[-1]
    dt_scaled = t_f / N
    res = np.zeros(6 * N)
    
    for i in range(N):
        # Midpoints for Hermite-Simpson method
        um = 0.5 * (u[i] + u[i+1]) + (dt_scaled / 8) * (c1 * U1[i] * np.sin(theta[i]) / m[i] - c1 * U1[i+1] * np.sin(theta[i+1]) / m[i+1])
        vm = 0.5 * (v[i] + v[i+1]) + (dt_scaled / 8) * (c1 * U1[i] * np.cos(theta[i]) / m[i] - g - (c1 * U1[i+1] * np.cos(theta[i+1]) / m[i+1] - g))
        thetam = 0.5 * (theta[i] + theta[i+1]) + (dt_scaled / 8) * (c3 * U2[i] - c3 * U2[i+1])
        mm = 0.5 * (m[i] + m[i+1]) + (dt_scaled / 8) * (-c1 * U1[i] / c2 - (-c1 * U1[i+1] / c2))
        U1m = 0.5 * (U1[i] + U1[i+1])
        U2m = 0.5 * (U2[i] + U2[i+1])
        
        # Hermite-Simpson collocation
        res[i] = x[i+1] - x[i] - (dt_scaled / 6) * (u[i] + 4*um + u[i+1])
        res[N+i] = y[i+1] - y[i] - (dt_scaled / 6) * (v[i] + 4*vm + v[i+1])
        res[2*N+i] = u[i+1] - u[i] - (dt_scaled / 6) * ((c1 * U1[i] * np.sin(theta[i]) / m[i]) + 4 * (c1 * U1m * np.sin(thetam) / mm) + (c1 * U1[i+1] * np.sin(theta[i+1]) / m[i+1]))
        res[3*N+i] = v[i+1] - v[i] - (dt_scaled / 6) * ((c1 * U1[i] * np.cos(theta[i]) / m[i] - g) + 4 * (c1 * U1m * np.cos(thetam) / mm - g) + (c1 * U1[i+1] * np.cos(theta[i+1]) / m[i+1] - g))
        res[4*N+i] = theta[i+1] - theta[i] - (dt_scaled / 6) * (c3 * U2[i] + 4 * c3 * U2m + c3 * U2[i+1])
        res[5*N+i] = m[i+1] - m[i] - (dt_scaled / 6) * ((-c1 * U1[i] / c2) + 4 * (-c1 * U1m / c2) + (-c1 * U1[i+1] / c2))
    
    return res
