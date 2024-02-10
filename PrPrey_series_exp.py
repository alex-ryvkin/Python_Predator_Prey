from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from itertools import combinations
#########################
Nexp = 3
x0 = 10
y0 = 10

########################
timestep = 0.01
end_time = 10

def PrPr(Listparam):
    pars = ['a', 'b', 'c', 'd', 'D1', 'D2']
    x = [x0]
    y = [y0]
    t = np.arange(0, end_time, timestep)
    R = []
    Dparam = dict(zip(pars, Listparam))
    print('----',Dparam )
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



path = r"C:\\Users\\Alexander\\PycharmProjects\\SLAVA\\PrPrOUT\\"
SER_csv =path+ 'series.csv'
pars =['a', 'b','c','d','D1','D2']
init_par = [1,0.1,0.5,0.02,0.01,0.01]


pairs = list(combinations(pars, 2))
random.shuffle(pairs)
series = ['ser','par1','par2']
seriesM =[]
#for ser in range(0, len(pairs)):
CSV_par =path+ 'parameters.csv'
colParams = ['series', 'exp', 'unique name', 'a', 'b', 'c', 'd', 'D1', 'D2']
PAR =[]
for ser in range(0, 3):
    DF = pd.DataFrame([init_par], columns=pars)
    seriesM.append([ser,pairs[ser][0],pairs[ser][1]])

    for exper in range (Nexp):

        unique = str(random.randint(100000, 999999))+'.csv'

        DF[pairs[ser][0]] = round(random.uniform(0, 0.05), 5)
        DF[pairs[ser][1]] = round(random.uniform(0, 0.05), 5)

        PAR.append([ser, exper, unique] + DF.iloc[0].tolist())    #????
        #print(DF.iloc[0].tolist())
        DFRES=PrPr(DF.iloc[0].tolist())

        #PrPr
        unique= path+unique
        DFRES.to_csv(unique, index=False)
dfPar = pd.DataFrame(PAR, columns=colParams)
dfPar.to_csv(CSV_par, index=False)

dfS = pd.DataFrame(seriesM, columns=series)
dfS.to_csv(SER_csv, index=False)






