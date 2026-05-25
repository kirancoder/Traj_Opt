import jax.numpy as jnp
from config import (
    N,
    c1,
    c2,
    m,
    g,
    x_0,
    y_0,
    u_0,
    v_0,
    theta_0,
    x_f,
    y_f,
    u_f,
    v_f,
    theta_f,
)


# Objective function: Qudratic cost on control derivatives
def objective_quad(vars):
    t_f = vars[-1]
    U1 = vars[5 * (N + 1) : 6 * (N + 1)]
    U2 = vars[6 * (N + 1) : 7 * (N + 1)]

    U1_dot = jnp.zeros(N + 1)
    U2_dot = jnp.zeros(N + 1)

    U1_dot = jnp.concatenate(
        [
            jnp.array([(-3 * U1[0] + 4 * U1[1] - U1[2]) / 2]),
            (U1[2:] - U1[:-2]) / 2,
            jnp.array([(3 * U1[-1] - 4 * U1[-2] + U1[-3]) / 2]),
        ]
    )

    U2_dot = jnp.concatenate(
        [
            jnp.array([(-3 * U2[0] + 4 * U2[1] - U2[2]) / 2]),
            (U2[2:] - U2[:-2]) / 2,
            jnp.array([(3 * U2[-1] - 4 * U2[-2] + U2[-3]) / 2]),
        ]
    )

    U_dot_norm_sq = jnp.sum(U1_dot**2) + jnp.sum(U2_dot**2)

    return t_f + U_dot_norm_sq


# time optimization objective
def objective_time(vars):
    return vars[-1]


# Define the dynamics constraints using Hermite-Simpson method
def dynamics(vars):
    u = vars[: N + 1]
    v = vars[N + 1 : 2 * (N + 1)]
    x = vars[2 * (N + 1) : 3 * (N + 1)]
    y = vars[3 * (N + 1) : 4 * (N + 1)]
    theta = vars[4 * (N + 1) : 5 * (N + 1)]
    U1 = vars[5 * (N + 1) : 6 * (N + 1)]
    U2 = vars[6 * (N + 1) : 7 * (N + 1)]
    t_f = vars[-1]
    dt_scaled = t_f / N
    res = jnp.zeros(5 * N)

    for i in range(N):
        um = (
            0.5 * (u[i] + u[i + 1])
            + (dt_scaled / 8)
            * (c1 * U1[i] * jnp.sin(theta[i]) - c1 * U1[i + 1] * jnp.sin(theta[i + 1]))
            / m
        )
        vm = (
            0.5 * (v[i] + v[i + 1])
            + (dt_scaled / 8)
            * (c1 * U1[i] * jnp.cos(theta[i]) - c1 * U1[i + 1] * jnp.cos(theta[i + 1]))
            / m
        )
        xm = 0.5 * (x[i] + x[i + 1]) + (dt_scaled / 8) * (u[i] - u[i + 1])
        ym = 0.5 * (y[i] + y[i + 1]) + (dt_scaled / 8) * (v[i] - v[i + 1])
        thetam = 0.5 * (theta[i] + theta[i + 1]) + (dt_scaled / 8) * (U2[i] - U2[i + 1])
        Um1 = 0.5 * (U1[i] + U1[i + 1])
        Um2 = 0.5 * (U2[i] + U2[i + 1])

        res = res.at[i].set(
            x[i + 1] - x[i] - (dt_scaled / 6) * (u[i] + 4 * um + u[i + 1])
        )
        res = res.at[N + i].set(
            y[i + 1] - y[i] - (dt_scaled / 6) * (v[i] + 4 * vm + v[i + 1])
        )
        res = res.at[2 * N + i].set(
            u[i + 1]
            - u[i]
            - (dt_scaled / 6)
            * c1
            * (
                U1[i] * jnp.sin(theta[i])
                + 4 * Um1 * jnp.sin(thetam)
                + U1[i + 1] * jnp.sin(theta[i + 1])
            )
            / m
        )
        res = res.at[3 * N + i].set(
            v[i + 1]
            - v[i]
            - (dt_scaled / 6)
            * (
                c1
                * (
                    U1[i] * jnp.cos(theta[i])
                    + 4 * Um1 * jnp.cos(thetam)
                    + U1[i + 1] * jnp.cos(theta[i + 1])
                )
                / m
                - g
            )
        )
        res = res.at[4 * N + i].set(
            theta[i + 1]
            - theta[i]
            - (dt_scaled / 6) * c2 * (U2[i] + 4 * Um2 + U2[i + 1])
        )

    return res


# Define boundary conditions
def boundary_conditions(vars):
    u = vars[: N + 1]
    v = vars[N + 1 : 2 * (N + 1)]
    x = vars[2 * (N + 1) : 3 * (N + 1)]
    y = vars[3 * (N + 1) : 4 * (N + 1)]
    theta = vars[4 * (N + 1) : 5 * (N + 1)]

    bc = jnp.zeros(10)
    bc = bc.at[0].set(x[0] - x_0)
    bc = bc.at[1].set(y[0] - y_0)
    bc = bc.at[2].set(u[0] - u_0)
    bc = bc.at[3].set(v[0] - v_0)
    bc = bc.at[4].set(theta[0] - theta_0)
    bc = bc.at[5].set(x[-1] - x_f)
    bc = bc.at[6].set(y[-1] - y_f)
    bc = bc.at[7].set(u[-1] - u_f)
    bc = bc.at[8].set(v[-1] - v_f)
    bc = bc.at[9].set(theta[-1] - theta_f)

    return bc
