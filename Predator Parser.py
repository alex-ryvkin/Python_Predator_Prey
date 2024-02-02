# import numpy as np
# import pandas as pd
# from scipy.integrate import odeint
# from scipy.optimize import minimize
#
# # Define the Lotka-Volterra equations
# def predator_prey_system(y, t, alpha, beta, delta, gamma):
#     prey, predator = y
#     dprey_dt = alpha * prey - beta * prey * predator
#     dpredator_dt = -delta * predator + gamma * prey * predator
#     return [dprey_dt, dpredator_dt]
#
# # Load the data from the CSV file
# data = pd.read_csv('predator_prey_results.csv')
# time_data = data['Time']
# prey_data = data['Prey Population']
# predator_data = data['Predator Population']
#
# # Define the objective function to minimize (sum of squared differences)
# def objective(params):
#     alpha, beta, delta, gamma = params
#     initial_population = [prey_data[0], predator_data[0]]
#     simulated_data = odeint(predator_prey_system, initial_population, time_data, args=(alpha, beta, delta, gamma))
#     prey_simulated, predator_simulated = simulated_data.T
#     error = np.sum((prey_simulated - prey_data)**2 + (predator_simulated - predator_data)**2)
#     return error
#
# # Set bounds for parameter values (you can adjust these)
# bounds = [(0, 2), (0, 2), (0, 2), (0, 2)]
#
# # Initial guess for parameters
# initial_guess = [0.5, 0.1, 0.5, 0.1]
#
# # Perform PSO optimization
# result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
#
# # Extract the optimized parameters
# optimal_params = result.x
# alpha_optimal, beta_optimal, delta_optimal, gamma_optimal = optimal_params
#
# print("Optimal Parameters:")
# print(f"Alpha: {alpha_optimal}")
# print(f"Beta: {beta_optimal}")
# print(f"Delta: {delta_optimal}")
# print(f"Gamma: {gamma_optimal}")


###########################################################################################

import numpy as np
import random
import csv


# Define the Lotka-Volterra equations
def predator_prey_system(y, t, alpha, beta, delta, gamma):
    prey, predator = y
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = -delta * predator + gamma * prey * predator
    return [dprey_dt, dpredator_dt]


# Load the data from the CSV file
def load_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            data.append([float(row[0]), float(row[1]), float(row[2])])
    return np.array(data)


# Objective function to minimize the error between model and data
def objective_function(params, data):
    alpha, beta, delta, gamma = params
    initial_population = [data[0, 1], data[0, 2]]
    time_points = data[:, 0]
    simulated_data = np.zeros(data.shape)

    for i in range(len(time_points)):
        time_span = time_points[i]
        num_time_points = 1  # Use a single time point to compute the model's state
        solution = odeint(predator_prey_system, initial_population, [0, time_span], args=(alpha, beta, delta, gamma))
        simulated_data[i, 1] = solution[1, 0]  # Prey
        simulated_data[i, 2] = solution[1, 1]  # Predator

    error = np.sum((data[:, 1:] - simulated_data[:, 1:]) ** 2)
    return error


# Particle Swarm Optimization (PSO)
def pso(objective_function, data, num_particles, num_iterations):
    # Define PSO parameters
    num_dimensions = 4  # Number of parameters (alpha, beta, delta, gamma)
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive parameter
    c2 = 1.5  # Social parameter
    v_max = 0.2  # Maximum velocity

    # Initialize particle positions and velocities
    particles = np.random.rand(num_particles, num_dimensions)
    velocities = np.random.rand(num_particles, num_dimensions) * v_max
    pbest_positions = particles.copy()
    pbest_values = np.zeros(num_particles)
    gbest_position = None
    gbest_value = np.inf

    for i in range(num_iterations):
        for j in range(num_particles):
            current_position = particles[j]
            current_value = objective_function(current_position, data)

            if current_value < pbest_values[j]:
                pbest_values[j] = current_value
                pbest_positions[j] = current_position

            if current_value < gbest_value:
                gbest_value = current_value
                gbest_position = current_position

        for j in range(num_particles):
            velocities[j] = w * velocities[j] + c1 * random.random() * (
                        pbest_positions[j] - particles[j]) + c2 * random.random() * (gbest_position - particles[j])
            particles[j] = particles[j] + velocities[j]

    return gbest_position


# Load the data from the CSV file
data = load_data('predator_prey_results.csv')

# Set PSO parameters
num_particles = 30
num_iterations = 100

# Run PSO to optimize the parameters
best_params = pso(objective_function, data, num_particles, num_iterations)

# Print the best-fit parameters
alpha, beta, delta, gamma = best_params
print(f'Optimal Parameters:')
print(f'Alpha: {alpha}')
print(f'Beta: {beta}')
print(f'Delta: {delta}')
print(f'Gamma: {gamma}')
