import matplotlib.pyplot as plt
# from config import N

def plot_trajectory(optimal_vars, N, title="Trajectory"):

    # Extract results
    # optimal_vars = result.x
    # optimal_t_f = optimal_vars[-1]
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(optimal_vars[:N+1], label='u')
    plt.plot(optimal_vars[N+1:2*(N+1)], label='v')
    plt.plot(optimal_vars[2*(N+1):3*(N+1)], label='x')
    plt.plot(optimal_vars[3*(N+1):4*(N+1)], label='y')
    plt.plot(optimal_vars[4*(N+1):5*(N+1)], label='theta')
    plt.plot(optimal_vars[5*(N+1):6*(N+1)], label='U1')
    plt.plot(optimal_vars[6*(N+1):7*(N+1)], label='U2')
    plt.xlabel('Time')
    plt.ylabel('State/Control Variables')
    plt.legend()
    plt.title('Optimal Trajectories')
    plt.show()