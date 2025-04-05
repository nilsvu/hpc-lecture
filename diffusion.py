from mpi4py import MPI
import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def take_timestep(u_next, u, alpha, dx, dt):
    for i in numba.prange(1, len(u) - 1):
        u_next[i] = u[i] + alpha * dt / dx**2 * (u[i + 1] - 2 * u[i] + u[i - 1])


def initial_data(x):
    # Gaussian blob in the center
    return np.exp(-100 * (x - 0.5) ** 2)


if __name__ == "__main__":
    # Simulation parameters
    total_num_points = 100
    num_time_steps = 100
    domain_length = 1.0
    dx = domain_length / total_num_points
    alpha = 0.01  # Thermal diffusivity
    dt = 0.4 * dx**2 / alpha  # CFL condition

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    # Distribution of points to processes
    if total_num_points % num_ranks != 0:
        if rank == 0:
            print("Number of points must be divisible by number of processes.")
        MPI.Finalize()
        exit(1)
    local_num_points = total_num_points // num_ranks
    if rank == 0:
        print(f"Using {num_ranks} processes with {local_num_points} points each.")
    index_start = rank * local_num_points
    index_end = index_start + local_num_points

    # Define coordinates with ghost zones
    # [ _____ | u[1]  | ... | u[-2] | _____ ]
    # index_start ^             ^ index_end
    x = np.linspace((index_start - 1) * dx, (index_end + 1) * dx, local_num_points + 2)

    # Set initial data
    u = initial_data(x)
    u_next = np.zeros_like(u)

    # Warm up JIT compiler
    take_timestep(u_next, u, alpha, dx, dt)

    # Simulation loop
    for t in range(num_time_steps):
        # Communicate ghost zones:
        #     [ _____ | u[1]  | ...
        # ... | u[-2] | _____ ]
        if rank > 0:
            comm.Send(u[1:2], dest=rank - 1)
            comm.Recv(u[0:1], source=rank - 1)
        if rank < num_ranks - 1:
            comm.Recv(u[-1:], source=rank + 1)
            comm.Send(u[-2:-1], dest=rank + 1)

        # Take time step
        take_timestep(u_next, u, alpha, dx, dt)
        u[:] = u_next[:]

        if rank == 0 and (t + 1) % 10 == 0:
            print(f"Completed time step {t + 1}/{num_time_steps}")

    # Collect results on rank 0
    u_all = None
    if rank == 0:
        u_all = np.zeros(total_num_points)
    comm.Gather(u[1:-1], u_all, root=0)

    # Plot result
    if rank == 0:
        import matplotlib.pyplot as plt

        x_all = np.linspace(0, domain_length, total_num_points)
        plt.plot(x_all, initial_data(x_all), color="black", label="Initial data", lw=2)
        plt.plot(
            x_all,
            u_all,
            color="red",
            label=f"Soluton at t = {num_time_steps * dt:.2f}",
            lw=2,
        )
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend()
        plt.grid()
        plt.show()
