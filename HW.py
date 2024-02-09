import numpy as npy
import random
import pandas as pd
from itertools import combinations

def pred_prey():
    x=[]
    y=[]

    for index in range(1, len(t)):
        # evaluate the current differentials
        xd = x[index - 1] * (a - b * y[index - 1]) - D1 * x[index - 1] ** 2
        yd = -y[index - 1] * (c - d * x[index - 1]) - D2 * y[index - 1] ** 2

        # evaluate the next value of x and y using differentials
        next_x = x[index - 1] + xd * timestep
        next_y = y[index - 1] + yd * timestep

        # add the next value of x and y
        x.append(next_x)
        y.append(next_y)
        RES.append([index * timestep, next_x, next_y])
    return [x,y]


# timestep determines the accuracy of the euler method of integration
timestep = 0.01
# amplitude of noise term
amp = 0.00
# the time at which the simulation ends
end_time = 200
############################################################################
Nexp = 25
Nser = 3
SER_csv = 'series.csv'

CSV_par = 'parameters.csv'
pars =['a', 'b','c','d','D1','D2']
pairs = list(combinations(pars, 2))
random.shuffle(pairs)
for ns in range(Nser):
    init_par = [1,0.1,0.5,0.02,0.01,0.01]
    limits = [1.5,0.5,1,0.1,0.05,0.05]
    data = [init_par, limits]
    DF = pd.DataFrame(data, columns=pars)
    for i in range(Nexp):

        DF[pairs[i][0]][0]=round(random.uniform(0, DF[pairs[0][0]])[1],5)
        DF[pairs[i][1]][0]=round(random.uniform(0, DF[pairs[0][1]])[1],5)




