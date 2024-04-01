import csv
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from model.model_updated_noise import model

t_final = 600.0
tsteps = int(t_final) + 1
t = np.linspace(0, 600, 601) # min
conversion = [0]

def pid2(sp,pv,pv_last,ierr,dt):
    # Parameters in terms of PID coefficients
    if heater == 1:
        Kc = 1/Kp1
    else:
        Kc = 1/(0.5 * Kp1)


    # Parameters in terms of PID coefficients
    KP = Kc
    KI = Kc/tau_i
    KD = Kc*tau_d
    # ubias for controller (initial heater)
    op0 = 0
    # upper and lower bounds on heater level
    if heater == 1:
        ophi = 100
    else:
        ophi = 75
    oplo = 0
    # calculate the error
    error = sp-pv
    # calculate the integral error
    ierr = ierr + KI * error * dt
    # calculate the measurement derivative
    dpv = (pv - pv_last) / dt
    # calculate the PID output
    P = KP * error
    I = ierr
    D = -KD * dpv
    op = op0 + P + I + D
    # implement anti-reset windup
    if op < oplo or op > ophi:
        I = I - KI * error * dt
        # clip output
        op = max(oplo,min(ophi,op))
    # return the controller output and PID terms
    return [op,P,I,D]
