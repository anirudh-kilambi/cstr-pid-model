import csv
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from model_updated_noise import model

# define initial arrays
Ca, Cc, T, Tc, qc = np.ones(301), np.ones(301), np.ones(301), np.ones(301), np.ones(301) * 0

# define initial parameters
Cao = 100 # mol/m^3
t_final = 300
tsteps = int(t_final) + 1
t = np.linspace(0, t_final, tsteps) # min
dt = t[2]-t[1]
Cco = 0 # mol/m^3
To = 298 # K
Tco = To # K
yo = [Cao, Cco, To, Tco]
q = 0.1

def conversion(Ca):
    return (Cao - Ca)/Cao

U = np.zeros(301)
Y = []

#setpoints
sp1_conversion = np.ones(301)
sp1_conversion[0:100] = 0.9
sp1_conversion[101:200] = 0.89
sp1_conversion[201:] = 0.915

sp2_qc = np.ones(301)

pv1_conversion = np.ones(301)
pv2_qc = np.ones(301)

ie1 = 0
ie2 = 0

#step test params
# Kc = 
# tau_i = 
# tau_d = 

#controller params
#outer
KP1 = -0.0435
KI1 = 0 #placeholder
KD1 = 0 #placeholder
#inner
KP2 = 1
KI2 = 0 #placeholder
KD2 = 0 #placeholder

def outer_pid(sp1, pv1, pv1_last, ierr, dt):
    """
    The outer controller that monitors the outlet concentration.
    Set point = conversion (X) via concentration
    Error = conversion setpoint - current conversion
    Outputs => coolant flowrate setpoint

    PARAMETERS
    ----------
    sp1 : current set point of conversion
    pv1 : current conversion
    pv_last1 : last measured conversion
    ierr1 : total sum integral error  
    """
    # Parameters in terms of PID coefficients
    # KP = Kc
    # KI = Kc/tau_i
    # KD = Kc*tau_d
    KC = 1/KP1
    KI = KI1
    KD = KD1
    # ubias for controller (initial heater)
    op0 = 0
    # upper and lower bounds on heater level
    ophi = 1
    oplo = 0
    # calculate the error
    error = sp1-pv1
    # calculate the integral error
    ierr += KI * error * dt
    # calculate the measurement derivative
    dpv = (pv1 - pv1_last) / dt
    # calculate the PID output
    P = KC * error
    I = ierr
    D = -KD * dpv
    op = op0 + P + I + D
    # implement anti-reset windup
    if np.all(op < oplo):
        I = I - KI * error * dt
        # op = max(oplo,min(ophi,op))
    # return the controller output and PID terms
    if np.all(op > ophi):
        I = I - KI * error * dt
    op = max(oplo,min(ophi,op))
    # return the controller output and PID terms
    return [op, I]

    # if op > ophi:
    #     I = I - KI * error * dt
    #     # clip output
    #     op = max(oplo,min(ophi,op))
    # # return the controller output and PID terms
    # return op

def inner_pid(sp2,pv2,pv_last2,ierr2,dt): 
    """
    The inner controller that handles the coolant flowrate.
    Set point = coolant flowrate (qc)
    Error = Coolant Flowrate Setpoint - Current Coolant Flowrate
    Outputs => valve actuation

    PARAMETERS
    ----------
    sp2 : current set point of coolant flowrate
    pv2 : current coolant flowrate
    pv_last2 : last measured coolant flowrate
    ierr2 : total sum integral error  
    """
    # Parameters in terms of PID coefficients
    KP = KP2
    KI = KI2
    KD = KD2
    # ubias for controller (initial heater)
    op0 = 0
    # upper and lower bounds on heater level
    ophi = 0.1
    oplo = 0
    # calculate the error
    error = sp2-pv2
    # calculate the integral error
    ierr2 += ierr2 + KI * error * dt
    # calculate the measurement derivative
    dpv = (pv2 - pv_last2) / dt
    # calculate the PID output
    P = KP * error
    I = ierr2
    D = -KD * dpv
    op = op0 + P + I + D
    # implement anti-reset windup
    if op < oplo:
        I = I - KI * error * dt
    if op > ophi:
        I = I - KI * error * dt
    op = max(oplo,min(ophi,op))
    # return the controller output and PID terms
    return [op, I]

for i in range(len(t)-1):
    ts = [t[i], t[i + 1]]
    if i < 1:
        sp2_qc[i], ie1 = outer_pid(sp1_conversion[i], pv1_conversion[i], 0, ie1, dt)
        U[i+1], ie2 = inner_pid(sp2_qc[i], pv2_qc[i], 0, ie2, dt)
    else:
        sp2_qc[i], ie1 = outer_pid(sp1_conversion[i], pv1_conversion[i], pv1_conversion[i-1], ie1, dt)
        U[i+1], ie2 = inner_pid(sp2_qc[i], pv2_qc[i], pv2_qc[i - 1], ie2, dt)

    qc[i] = U[i]
    y = odeint(model, yo, ts, args=(U[i], q), tfirst=True)
    yo = y[-1]
    Ca[i + 1], Cc[i + 1], T[i + 1], Tc[i + 1] = yo[0], yo[1], yo[2], yo[3]

for i in Ca:
    x = conversion(i)
    Y.append(x)





plt.figure()
plt.subplot(3,1,1)
plt.plot(t, Y, "b-", label="Conversion (X)")
plt.plot(t, sp1_conversion, "b:", label="Conversion SP")
plt.xlabel("time (min)")
plt.ylabel("conversion")
plt.legend()
plt.subplot(3,1,2)
plt.plot(t, qc, "r-")
plt.plot(t, sp2_qc, "r:", label="Coolant Flow Rate SP")
plt.xlabel("time (min)")
plt.ylabel("Coolant Flow Rate (m3/min)")
plt.legend()
plt.subplot(3,1,3)
plt.plot(t, U, label="Valve Open %")
plt.xlabel("time (min)")
plt.ylabel("Valve Open %")
plt.legend()
plt.show()