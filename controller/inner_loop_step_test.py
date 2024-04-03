import csv
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from model_updated_noise import model


t = np.linspace(0, 300, 301)  # t span, 600 minutes
conversion = [0]

# Step Test
u = np.ones(301) * 0
u[200:] = 0.5

qc = np.ones(301) * 0
R = 25
theta = np.sqrt(100/1.1)
Cv = 2
for i in range(len(t)):
    qc[i] = Cv * R**(u[i] - 1) * theta
    print(u[i], qc[i])

plt.figure()
plt.subplot(2,1,1)
plt.plot(t[100:400], qc[100:400], label="Coolant Flow Rate")
plt.legend()
plt.subplot(2,1,2)
plt.plot(t[100:400], u[100:400], label="Valve Position")
plt.legend()
plt.show()

