import numpy as np
import matplotlib.pyplot as plt
import numba
import time
import rich.progress


@numba.jit(nopython=True, parallel=True)
def compute_forces(positions, masses):
    num_particles = len(positions)
    forces = np.zeros((num_particles, 3))

    # Parallelize the outer loop over particles
    # for i in range(num_particles):
    for i in numba.prange(num_particles):
        # Add up forces from all other particles
        total_force = np.zeros(3)
        for j in range(num_particles):
            if i == j:
                continue
            r_ij = positions[j] - positions[i]
            distance = np.linalg.norm(r_ij)
            force_magnitude = masses[i] * masses[j] / distance**2
            # forces[i] += force_magnitude * (r_ij / distance)  # Possible race condition!
            total_force += force_magnitude * (r_ij / distance)
        forces[i] = total_force

    return forces


def take_timestep(positions, velocities, masses, dt):
    num_particles = len(positions)
    forces = compute_forces(positions, masses)
    for i in range(num_particles):
        accelerations = forces[i] / masses[i]  # F = ma
        velocities[i] += accelerations * dt
        positions[i] += velocities[i] * dt


def plot_trajectories(trajectories):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    num_particles = trajectories.shape[1]
    for i in range(num_particles):
        x, y, z = trajectories[:, i].T
        color = f"C{i}"
        ax.plot(x, y, z, lw=1, color=color)
        ax.scatter(x[-1], y[-1], z[-1], color=color)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    num_particles = 1000  # Cost grows with N^2
    num_timesteps = 200
    dt = 0.01

    # Initialize particles
    positions = (np.random.rand(num_particles, 3) * 2 - 1) * 10
    masses = np.random.rand(num_particles) * 1 + 0.1
    velocities = (np.random.rand(num_particles, 3) * 2 - 1) * 0.1

    # Array to store trajectories
    trajectories = np.zeros((num_timesteps, num_particles, 3))

    # Warm up the JIT compiler
    print("Warming up JIT compiler...")
    take_timestep(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), dt)

    # Simulation loop
    start_time = time.time()
    for t in rich.progress.track(range(num_timesteps), description="Simulating..."):
        take_timestep(positions, velocities, masses, dt)
        trajectories[t] = positions
    print(f"Simulation time: {time.time() - start_time:.2f} seconds")

    # Plot the trajectories
    plot_trajectories(trajectories)
