import matplotlib.pyplot as plt
import numpy as np

# a1 = 0.1
# b12 = 0.05#0.005#0.02
# b21= 0.04#0.01#0.1
# a2=0.2#0.1

# r (prey growth rate): 0.5
# a (predator death rate): 0.1
# b (predation rate): 0.3
# d (predator growth rate): 0.2
#
# Initial populations:
# N0 (initial prey population): 100
# P0 (initial predator population): 10
# r (скорость роста жертв): 0.6
# a (скорость смерти хищника): 0.2
# b (скорость хищничества): 0.8
# d (скорость роста хищника): 0.4
#
# Начальные популяции:
# N0 (начальная популяция жертв): 100
# P0 (начальная популяция хищников): 50

a1 = 0.1
b12 = 0.05

b21 = 0.5#0.03
a2 = 0.2

c1 = 0#0.1
c2 = 0#0.01


dt = 0.1
HH =[]
WW=[]

time = np.arange(0,2000,dt)

H= 100
W =20

for t in time:
    if(H<0):
        H=0
    if (W < 0):
        W = 0
    HH.append(H)

    WW.append(W)
    H+=dt*(a1*H-b12*H*W-c1*H)
    W+=dt*(b21*H*W-a2*W-c2*W)
print(len(time),len(HH))
plt.plot(time,HH, label='HH')
plt.plot(time,WW, label='WW')
plt.legend()
plt.show()

plt.plot(HH,WW, label='HH')
plt.xlabel('H')  # Adding x-axis label
plt.ylabel('W')  # Adding y-axis label

plt.legend()
plt.show()
