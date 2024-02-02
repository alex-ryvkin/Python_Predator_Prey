from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as npy
import random

# timestep determines the accuracy of the euler method of integration
timestep = 0.001
# amplitude of noise term
amp = 0.00
# the time at which the simulation ends
end_time = 1000

# creates a time vector from 0 to end_time, seperated by a timestep
t = npy.arange(0, end_time, timestep)

# intialize rabbits (x) and foxes (y) vectors
x = []
y = []


# noise term to perturb differential equations
def StochasticTerm(amp):
    return (amp * random.uniform(-1, 1))


"""" definition of lotka-volterra parameters"""
# birth rate of rabbits
a = 0.1
# death rate of rabbits due to predation
b = 0.1
# natural death rate of foxes
c = 0.05#0.5
# effectiveness of eating rabbits for fox`s birth
d = 0.005#0.02

""" euler integration """

# initial conditions for the rabbit (x) and fox (y) populations at time=0
x.append(100)
y.append(20)

# forward euler method of integration
# a perturbbation term is added to the differentials to make the simulation stochastic
for index in range(1, len(t)):
    # make parameters stochastic
    #     a = a + StochasticTerm(amp)
    #     b = b + StochasticTerm(amp)
    #     c = c + StochasticTerm(amp)
    #     d = d + StochasticTerm(amp)

    # evaluate the current differentials
    xd = x[index - 1] * (a - b * y[index - 1])
    yd = -y[index - 1] * (c - d * x[index - 1])

    # evaluate the next value of x and y using differentials
    next_x = x[index - 1] + xd * timestep
    next_y = y[index - 1] + yd * timestep

    # add the next value of x and y
    x.append(next_x)
    y.append(next_y)

""" visualization """

if amp == 0:
    # visualization of deterministic populations against time
    plt.plot(t, x)
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.legend(('Rabbits', 'Foxes'))
    plt.title('Deterministic Lotka-Volterra')

    plt.show()

# deterministic phase portrait
    plt.plot(x, y)
    plt.xlabel('Fox Population')
    plt.ylabel('Rabbit Population')
    plt.title('Phase Portrait of Deterministic Lotka-Volterra')

    plt.show()

else:
# visualization of stochastic populations against time
    plt.plot(t, x)
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.legend(('Rabbits', 'Foxes'))
    plt.title('Stochastic Lotka-Volterra')

    plt.show()

    # stochastic phase portrait
    plt.plot(x, y)
    plt.xlabel('Fox Population')
    plt.ylabel('Rabbit Population')
    plt.title('Phase Portrait of Stochastic Lotka-Volterra')

    plt.show()

# noise term visualization
noise = []
n = []
for sample in range(100):
    noise.append(StochasticTerm(amp))
    n.append(sample)

plt.plot(n, noise)
plt.xlabel('Arbitrary Noise Samples')
plt.ylabel('Noise')
plt.title('Perturbation to Birth Rate')

plt.show()