from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as npy
import random
import pandas as pd

# timestep determines the accuracy of the euler method of integration
timestep = 0.0001
# amplitude of noise term
amp = 0.00
# the time at which the simulation ends
end_time = 50

t = npy.arange(0, end_time, timestep)

# intialize rabbits (x) and foxes (y) vectors
x = []
y = []

RES = []

CSV_dum = 'Dumped2.csv'

# birth rate of rabbits
a = 1
# death rate of rabbits due to predation
b = 0.1
# natural death rate of foxes
c = 0.5
# factor that describes how many eaten rabbits give birth to a new fox
d = 0.02
# concurence factor of prey-prey
D1 = 0.01
# concurence factor of pred-pred
D2 = 1

x.append(100)
y.append(20)


for index in range(1, len(t)):
    # make parameters stochastic
    #     a = a + StochasticTerm(amp)
    #     b = b + StochasticTerm(amp)
    #     c = c + StochasticTerm(amp)
    #     d = d + StochasticTerm(amp)

    # evaluate the current differentials
    # xd = x[index - 1] * (a - b * y[index - 1])
    # yd = -y[index - 1] * (c - d * x[index - 1])
    #MODIFIED
    xd = x[index - 1] * (a - b * y[index - 1])-D1*x[index - 1]**2
    yd = -y[index - 1] * (c - d * x[index - 1])-D2*y[index - 1]**2



    # evaluate the next value of x and y using differentials
    next_x = x[index - 1] + xd * timestep
    next_y = y[index - 1] + yd * timestep

    # add the next value of x and y
    x.append(next_x)
    y.append(next_y)

    RES.append([index * timestep, next_x, next_y])

columns_res = ['t','R','F']
df = pd.DataFrame(RES, columns=columns_res)
df.to_csv(CSV_dum, index=False)
""" visualization """

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



