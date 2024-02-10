from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from itertools import combinations

timestep = 0.01
end_time = 500

def PrPr(Listparam):
    pars = ['a', 'b', 'c', 'd', 'D1', 'D2']
    x = [10]
    y = [10]
    t = np.arange(0, end_time, timestep)
    R = []
    Dparam = dict(zip(pars, Listparam))
    for i in range(len(Listparam)):
        pars[i] = Listparam[i]
    #init cond
    for index in range(1, len(t)):

        # evaluate the current differentials
        xd = x[index - 1] * (Dparam['a'] - Dparam['b'] * y[index - 1])-Dparam['D1']*x[index - 1]**2
        yd = -y[index - 1] * (Dparam['c'] - Dparam['d'] * x[index - 1])-Dparam['D2']*y[index - 1]**2

        # evaluate the next value of x and y using differentials
        next_x = x[index - 1] + xd * timestep
        next_y = y[index - 1] + yd * timestep

        # add the next value of x and y
        x.append(next_x)
        y.append(next_y)
        R.append([index*timestep, next_x, next_y])
    columns_res = ['time', 'x', 'y']
    DF_prpr = pd.DataFrame(R, columns=columns_res)
    return DF_prpr

Listparam = [1, 0.1, 0.5, 0.02, 0.01, 0.01]
print(PrPr(Listparam))