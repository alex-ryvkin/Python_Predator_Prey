
##
import numpy as np

# Define the Himmelblau function
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Particle swarm optimization function
def particle_swarm_optimization(obj_func, bounds, num_particles, num_iterations, inertia_weight, cognitive_weight, social_weight):
    # Initialize particles' positions and velocities within the bounds
    dimensions = len(bounds)
    particle_positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, dimensions))
    particle_velocities = np.zeros((num_particles, dimensions))

    # Initialize the best known positions and fitness values for each particle
    particle_best_positions = particle_positions.copy()
    particle_best_fitness = np.full(num_particles, np.inf)

    # Initialize the global best position and fitness value
    global_best_position = np.zeros(dimensions)
    global_best_fitness = np.inf

    for _ in range(num_iterations):
        # Evaluate the fitness value for each particle
        particle_fitness = obj_func(particle_positions[:, 0], particle_positions[:, 1])

        # Update the best known positions and fitness values for each particle
        update_indices = particle_fitness < particle_best_fitness
        particle_best_positions[update_indices] = particle_positions[update_indices]
        particle_best_fitness[update_indices] = particle_fitness[update_indices]

        # Update the global best position and fitness value
        best_particle_index = np.argmin(particle_best_fitness)
        if particle_best_fitness[best_particle_index] < global_best_fitness:
            global_best_position = particle_best_positions[best_particle_index].copy()
            global_best_fitness = particle_best_fitness[best_particle_index]

        # Update the particle velocities and positions
        r1 = np.random.random((num_particles, dimensions))
        r2 = np.random.random((num_particles, dimensions))
        particle_velocities = (inertia_weight * particle_velocities
                               + cognitive_weight * r1 * (particle_best_positions - particle_positions)
                               + social_weight * r2 * (global_best_position - particle_positions))
        particle_positions += particle_velocities

        # Apply boundary constraints
        particle_positions = np.clip(particle_positions, bounds[:, 0], bounds[:, 1])

    return global_best_position, global_best_fitness

# Define the bounds of the search space
bounds = np.array([[-5, 5], [-5, 5]])

# Set the number of particles and iterations
num_particles = 10
num_iterations = 100

# Set the inertia weight, cognitive weight, and social weight
inertia_weight = 0.7
cognitive_weight = 1.4
social_weight = 1.4

# Run the particle swarm optimization
best_position, best_fitness = particle_swarm_optimization(himmelblau, bounds, num_particles, num_iterations,
                                                          inertia_weight, cognitive_weight, social_weight)

# Print the results
print("Best position:", best_position)
print("Best fitness value:", best_fitness)
