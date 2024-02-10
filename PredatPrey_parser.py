import matplotlib.pyplot as plt
import numpy as npy
import random
import pandas as pd

def local_min(t,Rab):
    loc_min=[]
    for i in range (1,len(Rab)-1):
        if(Rab[i-1]>Rab[i] and Rab[i]<Rab[i+1]):
            loc_min.append([t[i],Rab[i]])
    return loc_min

Dum1=[] #[[D1,D2]]
Dum2=[] #[[D1,D2]]
Stat1=[]
Stat2=[]


path = r"C:\\Users\\Alexander\\PycharmProjects\\SLAVA\\PrPrOUT\\"
SER_csv =path+ 'series.csv'
DFs = pd.read_csv(SER_csv)             #, index_col=None?????
SER_csv =path+ 'parameters.csv'
DFp = pd.read_csv(SER_csv)             #,  index_col=None)

# FilDFs = DFs[DFs['ser']==1]
# print(FilDFs)
    # print(ss)
    # SER=DFs.iloc[ss].tolist()
    # print(SER)
    # i = SER[0]
    # Lpar = [SER[1],SER[2]]
for pp in range(len(DFp)):
    ns = int(DFp['series'][pp])

    FilDFs = DFs[DFs['ser']==ns]                  #automatic???

    Lpar = [FilDFs['par1'][ns], FilDFs['par2'][ns]]
    #Lpar = [FilDFs['par1'][ns], FilDFs['par2'][ns]]
    print(FilDFs['par1'][ns],"      ",FilDFs.iloc[0,1])





# DF = pd.read_csv("parameters.csv")
# for i in range(len(DF['index'])):
#     f = DF['index'][i]+'.csv'
#     file=pd.read_csv(f)
#     if len(local_min(file['t'],file['R']))>1:
#         Dum1.append(DF["D1"][i].tolist())
#         Dum2.append(DF["D2"][i].tolist())
#
#     else:
#         Stat1.append(DF["D1"][i].tolist())
#         Stat2.append(DF["D2"][i].tolist())
#
# plt.scatter(Dum1, Dum2, color='blue', marker='o', label='Dum')
#
# plt.scatter(Stat1, Stat2, color='red', marker='o', label='Dum')
#
# # Adding labels and title
# plt.xlabel('D1')
# plt.ylabel('D2')
#
#
# # Adding a legend
# plt.legend()
#
# # Show the plot
# plt.show()
