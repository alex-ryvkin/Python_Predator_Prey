
##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the Himmelblau function
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def ackley(x, y):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    term3 = np.exp(1) + 20
    return term1 + term2 + term3

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
        #print(particle_positions[:, 0], particle_positions[:, 1])
        #print(MSE(0.26503686,0.53622074))
        particle_fitness = obj_func(particle_positions[:, 0], particle_positions[:, 1])   #!!!!!
        #print(particle_fitness)
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
        print(global_best_position)



    return global_best_position, global_best_fitness

# Define the bounds of the search space
#=====================================================================================================

def MSE(a,b):
    n = len(a)
    MS=np.empty(n)
    #print(a,b)
    for i in range (n):
        al =a[i]
        be = b[i]

        x_values = PrePre(al,be)
        prey_mod = x_values
        squared_errors = np.square(prey - prey_mod)

        # Calculate mean squared error
        mean_squared_error = np.mean(squared_errors)
        MS[i] = mean_squared_error
    return MS#mean_squared_error
#==========================================================
def PrePre(al,be):
    alpha = al
    beta = be
    #alpha = param[0]
    #alpha = param[:, 0]
    #beta = param[1]
    #beta = param[:, 0]
    x = x0
    y = y0
    x_values = [x]
    y_values = [y]

    for _ in range(num_steps):
        dx_dt = alpha * x - beta * x * y
        dy_dt = delta * x * y - gamma * y
        x += dx_dt * dt
        y += dy_dt * dt
        x_values.append(x)
        y_values.append(y)

    return np.array(x_values) #, np.array(y_values)
#=====================PrePre
x0 = 10  # initial prey population
y0 = 5  # initial predator population
#alpha = 0.1  # prey birth rate
#beta = 0.02  # predation rate
gamma = 0.3  # predator death rate
delta = 0.01  # predator reproduction rate
dt = 0.01  # time step size
num_steps = 20000  # number of time steps
# alpha = 0.1  # prey birth rate
# beta = 0.02  # predation rate
bounds = np.array([[0, 1], [0, 1]])

# Set the number of particles and iterations
num_particles = 10
num_iterations = 100

# Set the inertia weight, cognitive weight, and social weight
inertia_weight = 0.7
cognitive_weight = 1.4
social_weight = 1.4

# Run the particle swarm optimization

data = pd.read_csv('PrePreSimple.csv')
time  = data['t'].values
prey = data['x'].values
predator = data['y'].values

best_position, best_fitness = particle_swarm_optimization(MSE, bounds, num_particles, num_iterations,
                                                          inertia_weight, cognitive_weight, social_weight)
# Print the results
print("Best position:", best_position)
print("Best fitness value:", best_fitness)

x_values = PrePre(best_position[0],best_position[1])

plt.plot(time,prey)
plt.plot(time,x_values)
plt.show()

