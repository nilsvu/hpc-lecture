#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <random>

void compute_forces(Kokkos::View<double **> forces, const Kokkos::View<double **> positions, const Kokkos::View<double *> masses, const int num_particles)
{
    // Parallelize the outer loop over particles
    Kokkos::parallel_for("ComputeForces", num_particles, KOKKOS_LAMBDA(int i) {
        // Add up forces from all other particles
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int j = 0; j < num_particles; ++j) {
            if (i == j) continue;
            double dx = positions(j,0) - positions(i,0);
            double dy = positions(j,1) - positions(i,1);
            double dz = positions(j,2) - positions(i,2);
            double r2 = dx*dx + dy*dy + dz*dz;
            double r  = sqrt(r2);
            double force_mag = masses(i) * masses(j) / r2;
            fx += force_mag * dx / r;
            fy += force_mag * dy / r;
            fz += force_mag * dz / r;
        }
        forces(i,0) = fx;
        forces(i,1) = fy;
        forces(i,2) = fz; });
}

void take_timestep(Kokkos::View<double **> positions, Kokkos::View<double **> velocities, const Kokkos::View<double *> masses, Kokkos::View<double **> forces, double dt, const int num_particles)
{
    compute_forces(forces, positions, masses, num_particles);

    Kokkos::parallel_for("TakeTimestep", num_particles, KOKKOS_LAMBDA(int i) {
        for (int d = 0; d < 3; ++d) {
            double acceleration = forces(i,d) / masses(i);
            velocities(i,d) += acceleration * dt;
            positions(i,d)  += velocities(i,d) * dt;
        } });
}

void save_trajectories(const std::vector<Kokkos::View<double **>> &trajectories, const std::string &filename)
{
    std::ofstream file(filename);
    for (size_t t = 0; t < trajectories.size(); ++t)
    {
        auto h_traj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), trajectories[t]);
        for (size_t i = 0; i < h_traj.extent(0); ++i)
        {
            file << t << "," << i << "," << h_traj(i, 0) << "," << h_traj(i, 1) << "," << h_traj(i, 2) << "\n";
        }
    }
    file.close();
    std::cout << "Trajectories saved to " << filename << std::endl;
}

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        const int num_particles = (argc > 1) ? std::stoi(argv[1]) : 1000;
        const int num_timesteps = 200;
        const double dt = 0.01;

        Kokkos::View<double **> positions("positions", num_particles, 3);
        Kokkos::View<double **> velocities("velocities", num_particles, 3);
        Kokkos::View<double *> masses("masses", num_particles);
        Kokkos::View<double **> forces("forces", num_particles, 3);

        // Initialize with random values on the host
        auto h_pos = Kokkos::create_mirror_view(positions);
        auto h_vel = Kokkos::create_mirror_view(velocities);
        auto h_mass = Kokkos::create_mirror_view(masses);

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> pos_dist(-10, 10);
        std::uniform_real_distribution<> vel_dist(-0.1, 0.1);
        std::uniform_real_distribution<> mass_dist(0.1, 1.1);

        for (int i = 0; i < num_particles; ++i)
        {
            for (int d = 0; d < 3; ++d)
            {
                h_pos(i, d) = pos_dist(gen);
                h_vel(i, d) = vel_dist(gen);
            }
            h_mass(i) = mass_dist(gen);
        }

        Kokkos::deep_copy(positions, h_pos);
        Kokkos::deep_copy(velocities, h_vel);
        Kokkos::deep_copy(masses, h_mass);

        // Warmup
        std::cout << "Warming up...\n";
        take_timestep(positions, velocities, masses, forces, dt, num_particles);

        // Main simulation
        std::cout << "Simulating...\n";
        std::vector<Kokkos::View<double **>> trajectories;
        trajectories.reserve(num_timesteps);

        auto start = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < num_timesteps; ++t)
        {
            take_timestep(positions, velocities, masses, forces, dt, num_particles);

            // Save a copy of the current position view
            Kokkos::View<double **> snapshot("snapshot", num_particles, 3);
            Kokkos::deep_copy(snapshot, positions);
            trajectories.push_back(snapshot);

            if (t % 10 == 0)
                std::cout << "Timestep " << t << "/" << num_timesteps << "\r" << std::flush;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "\nSimulation completed in "
                  << std::chrono::duration<double>(end - start).count() << " seconds\n";

        save_trajectories(trajectories, "trajectories.csv");
    }
    Kokkos::finalize();
    return 0;
}
