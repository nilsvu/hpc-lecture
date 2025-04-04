#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <random>

using Vec3 = Eigen::Vector3d;

void compute_forces(std::vector<Vec3> &forces, const std::vector<Vec3> &positions, const std::vector<double> &masses)
{
    const int num_particles = positions.size();
    // Parallelize the outer loop over particles
#pragma omp parallel for
    for (int i = 0; i < num_particles; ++i)
    {
        // Add up forces from all other particles
        Vec3 total_force = Vec3::Zero();
        for (int j = 0; j < num_particles; ++j)
        {
            if (i == j)
                continue;
            Vec3 r_ij = positions[j] - positions[i];
            double distance = r_ij.norm();
            double force_mag = masses[i] * masses[j] / (distance * distance);
            total_force += force_mag * (r_ij / distance);
        }
        forces[i] = total_force;
    }
}

void take_timestep(std::vector<Vec3> &positions, std::vector<Vec3> &velocities, const std::vector<double> &masses, std::vector<Vec3> &forces, double dt)
{
    const int num_particles = positions.size();
    compute_forces(forces, positions, masses);
    for (int i = 0; i < num_particles; ++i)
    {
        Vec3 acceleration = forces[i] / masses[i];
        velocities[i] += acceleration * dt;
        positions[i] += velocities[i] * dt;
    }
}

void save_trajectories(const std::vector<std::vector<Vec3>> &trajectories, const std::string &filename)
{
    std::ofstream file(filename);
    for (size_t t = 0; t < trajectories.size(); ++t)
    {
        for (size_t i = 0; i < trajectories[t].size(); ++i)
        {
            const auto &pos = trajectories[t][i];
            file << t << "," << i << "," << pos[0] << "," << pos[1] << "," << pos[2] << "\n";
        }
    }
    file.close();
    std::cout << "Trajectories saved to " << filename << std::endl;
}

int main(int argc, char *argv[])
{
    const int num_particles = (argc > 1) ? std::stoi(argv[1]) : 1000;
    const int num_timesteps = 200;
    const double dt = 0.01;

    // Random initial conditions
    std::vector<Vec3> positions(num_particles);
    std::vector<Vec3> velocities(num_particles);
    std::vector<double> masses(num_particles);
    std::vector<Vec3> forces(num_particles);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> mass_dist(0.1, 1.1);
    for (int i = 0; i < num_particles; ++i)
    {
        positions[i] = (Vec3::Random()) * 10.0;
        velocities[i] = Vec3::Random() * 0.1;
        masses[i] = mass_dist(gen);
    }

    std::vector<std::vector<Vec3>> trajectories;
    trajectories.reserve(num_timesteps);

    std::cout << "Warming up...\n";
    take_timestep(positions, velocities, masses, forces, dt);

    std::cout << "Simulating...\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_timesteps; ++t)
    {
        take_timestep(positions, velocities, masses, forces, dt);
        trajectories.push_back(positions);
        if (t % 10 == 0)
            std::cout << "Timestep " << t << "/" << num_timesteps << "\r" << std::flush;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nSimulation completed in " << elapsed.count() << " seconds\n";

    save_trajectories(trajectories, "trajectories.csv");

    return 0;
}