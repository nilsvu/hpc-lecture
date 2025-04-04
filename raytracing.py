import multiprocessing as mp
import numpy as np
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import rich.progress


def geodesic_equation(phi, y):
    u, v = y
    return np.array([v, 3 * u**2 - u])


def trace_ray(b):
    r0 = 10
    u0 = 1 / r0
    v0 = np.sqrt(1 / b**2 - u0**2 + 2 * u0**3)

    # Termination conditions
    max_radius = 1.1 * r0

    def event_escape(t, y):
        return y[0] - 1 / max_radius

    def event_horizon(t, y):
        return y[0] - 0.5

    event_escape.terminal = True
    event_horizon.terminal = True

    trajectory = solve_ivp(
        geodesic_equation,
        (0, 10),
        y0=[u0, v0],
        events=[event_escape, event_horizon],
        rtol=1e-12,
        atol=1e-12,
        t_eval=np.linspace(0, 10, 100),
    )
    return trajectory.t, 1 / trajectory.y[0]


def plot_trajectories(trajectories):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    cmap = plt.get_cmap("viridis")

    for i, trajectory in enumerate(trajectories):
        phi, r = trajectory
        escapes = r[-1] > 2.5
        ax.plot(
            r * np.cos(phi),
            r * np.sin(phi),
            lw=1,
            color=cmap(i / (len(trajectories) - 1)) if escapes else "black",
        )

    # Draw the horizon
    ax.add_artist(plt.Circle((0, 0), 2, color="black", fill=False, lw=1.5, ls="dashed"))
    plt.show()


if __name__ == "__main__":
    num_rays = 1000
    b_crit = 3 * np.sqrt(3)  # Critical impact parameter
    delta_b = 0.1 * b_crit
    impact_params = np.linspace(b_crit - delta_b, b_crit + delta_b, num_rays)

    # Serial computation:
    results = []
    for b in rich.progress.track(impact_params, description="Tracing rays..."):
        results.append(trace_ray(b))

    # Multiprocessing:
    start_time = time.time()
    with mp.Pool() as pool:
        results = pool.map(trace_ray, impact_params)
    print(f"Simulation time: {time.time() - start_time:.2f} seconds")

    # Plot the trajectories
    plot_trajectories(results)
