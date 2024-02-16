import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd


def local_min(Signal):
    loc_min=[]
    for i in range (1,len(Signal)-1):
        if(Signal[i-1][1]>Signal[i][1] and Signal[i][1]<Signal[i+1][1]):
            loc_min.append([Signal[i][0],Signal[i][1]])
    return loc_min







path = r'c:\\Users\\Alexander\\PycharmProjects\\SLAVA\\PrPrOUT\\'
SER_csv = path+'series.csv'
DFs = pd.read_csv(SER_csv)             #, index_col=None?????
Nser = len(DFs)
PAR_csv = path+'parameters.csv'
DFp = pd.read_csv(PAR_csv)             #,  index_col=None)

# FilDFs = DFs[DFs['ser']==1]
# print(FilDFs)
    # print(ss)
    # SER=DFs.iloc[ss].tolist()
    # print(SER)
    # i = SER[0]
    # Lpar = [SER[1],SER[2]]
count = 0
Dum1 = [[] for _ in range(Nser)]

Dum2 = [[] for _ in range(Nser)]
Stat1 = [[] for _ in range(Nser)]
Stat2 = [[] for _ in range(Nser)]
PAR =list(zip(DFs["par1"], DFs["par2"]))


for pp in range(len(DFp)):
    ns = int(DFp['series'][pp])   # num  of series

    uniq = DFp['unique'][pp]
    uniq = path+ uniq
    Signal = pd.read_csv(uniq)   # read each exper

    #FilDFs = DFs[DFs['ser']==ns]
    #automatic???

    #Lpar = [FilDFs['par1'][ns], FilDFs['par2'][ns]]
    #Lpar = [FilDFs['par1'][ns], FilDFs['par2'][ns]]
    # print(FilDFs['par1'][ns],"      ",FilDFs.iloc[0,1])



    #for exp in range(len(FilDFs)):
    SigList = Signal.values.tolist()

    if len(local_min(SigList)) > 1:
        Dum1[ns].append(DFp[PAR[ns][0]][pp].tolist())
        Dum2[ns].append(DFp[PAR[ns][1]][pp].tolist())

    else:
        Stat1[ns].append(DFp[PAR[ns][0]][pp].tolist())
        Stat2[ns].append(DFp[PAR[ns][1]][pp].tolist())
        #count += 1
print(Stat1)
for ns in range(Nser):
    plt.scatter(Dum1[ns], Dum2[ns], color='blue', marker='o', label='Dum')

    plt.scatter(Stat1[ns], Stat2[ns], color='red', marker='o', label='Dum')

    # Adding labels and title
    plt.xlabel(PAR[ns][0])
    plt.ylabel(PAR[ns][1])

    # Adding a legend
    plt.legend()

        # Show the plot
    plt.show()

