from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as npy
import random
import pandas as pd
from itertools import combinations

def PrPr(Listparam):
    #array time
    #DFparam take abcdD1D2
    #pars = ['a', 'b', 'c', 'd', 'D1', 'D2']
    #Listparam = [1, 0.1, 0.5, 0.02, 0.01, 0.01]
    a= Listparam[0]
    x =[]
    y=[]
    #init cond
    for index in range(1, len(t)):

        # evaluate the current differentials
        xd = x[index - 1] * (a - b * y[index - 1])-D1*x[index - 1]**2
        yd = -y[index - 1] * (c - d * x[index - 1])-D2*y[index - 1]**2

        # evaluate the next value of x and y using differentials
        next_x = x[index - 1] + xd * timestep
        next_y = y[index - 1] + yd * timestep

        # add the next value of x and y
        x.append(next_x)
        y.append(next_y)
        RES.append([index*timestep, next_x, next_y])
    df = pd.DataFrame(RES, columns=columns_res)
    return DF_prpr
