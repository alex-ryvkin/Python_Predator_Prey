import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def lotka_volterra(x0, y0, alpha, beta, gamma, delta, dt, num_steps):
    """
    Solve the Lotka-Volterra equations using Euler's method.

    Parameters:
        x0: Initial population of prey.
        y0: Initial population of predators.
        alpha: Prey birth rate.
        beta: Predation rate.
        gamma: Predator death rate.
        delta: Predator reproduction rate.
        dt: Time step size.
        num_steps: Number of time steps.

    Returns:
        Tuple of arrays (x_values, y_values) containing the populations of prey and predators
        over time.
    """
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

    return np.array(x_values), np.array(y_values)


# Parameters
x0 = 10  # initial prey population
y0 = 5  # initial predator population
alpha = 0.1  # prey birth rate
beta = 0.02  # predation rate
gamma = 0.3  # predator death rate
delta = 0.01  # predator reproduction rate
dt = 0.01  # time step size
num_steps = 20000  # number of time steps
time = np.arange(0,(num_steps+1)*dt,dt)

# Solve the equations
x_values, y_values = lotka_volterra(x0, y0, alpha, beta, gamma, delta, dt, num_steps)
print(len(time),len(x_values))
df = pd.DataFrame({'t': time, 'x': y_values, 'y': y_values})
df.to_csv('PrePreSimple.csv', index=False)
# Plot the results
plt.plot(time,x_values, label='Prey')
plt.plot(time,y_values, label='Predator')
plt.xlabel('Time Step')
plt.ylabel('Population')
plt.title('Lotka-Volterra Model')
plt.legend()
plt.show()
