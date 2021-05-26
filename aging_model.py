###############################################################################
# Empirical Aging Model (ESCA)                                                #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n_samples = 2500
n_cycles = 1000

# sample random variables
eksi1 = np.random.uniform(2.0, 300.0, n_samples)
eksi2 = np.random.uniform(2.0, 300.0, n_samples)
eksi3 = np.random.uniform(0.8, 1.0, n_samples)
eksi4 = np.random.uniform(0.4, 0.8, n_samples)
eksi5 = np.random.uniform(333.15, 363.15, n_samples)
eksi6 = np.random.uniform(0.4, 1.0, n_samples)

loss = np.zeros((n_cycles, n_samples))
N = np.linspace(0,40000, n_cycles).astype(int)

# empirical constants
C     = 1.001
V0    = 0.98
R     = 8.314
F     = 96485.3
b     = -0.330745112261277
c     = 0.0422225897309428
d     = -0.0252559223825606
e     = 0.152227535372688
f     = -0.200580393019062
beta  = 1.72956278312408
gamma = 0.0367292597803257
delta = 0.0366750190665531
eps   = -0.102617650203082
zeta  = 0.213272924462045
EaA   = 60500.9981580511
EaB   = -59281.5159646116
# dataset-specific constants
a     = -22163752.7431349
alpha = -2.25221786507963e-08

for i in range(n_samples):
    # test conditions
    t_UPL = eksi1[i]
    t_LPL = eksi2[i]
    V_UPL = eksi3[i]
    V_LPL = eksi4[i]
    T     = eksi5[i]
    RH    = eksi6[i]

    A = a*(1+b*RH)*(1+c*np.log(1+t_UPL))*(1+d*np.log(1+t_LPL))\
        *np.exp(-EaA/(R*T))*np.exp(e*F/(R*T)*(V_UPL - V0))\
        *np.exp(f*F/(R*T)*(V_LPL -V0))

    B = alpha*(1+beta*RH)*(1+gamma*np.log(1+t_UPL))*(1+delta*np.log(1+t_LPL))\
        *np.exp(-EaB/(R*T))*np.exp(eps*F/(R*T)*(V_UPL - V0))\
        *np.exp(zeta*F/(R*T)*(V_LPL -V0))

    for j in range(n_cycles):
        loss[j,i] = C - (1/B)*(np.arcsinh(np.sinh(B*(C-1))*N[j]**(A*B)))

np.savetxt('data/LF.csv', loss, delimiter=',')

# plot data 
for _ in range(n_samples):
    plt.plot(N[:],loss[:,_], 'k-')
plt.xlabel('$N$')
plt.ylabel('ESCA loss')
plt.show()

print(np.where(np.sign(loss)==-1)[1])
