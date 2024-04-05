import csv
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from model_updated_noise import model
import time

kc1_stab = []
kc2_stab = []

kc1_test_range = 0
kc2_test_range = range(-50,100,1)

# define initial arrays
def stability_analysis(KP2):
    # define initial arrays
    Ca, Cc, T, Tc, qc = np.ones(601), np.ones(601), np.ones(601), np.ones(601), np.ones(601) * 0

    # define controller parameters
    KP1 = -63
    KD1 = 0.31
    KD2 = -3.01

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

    U = np.zeros(601)
    Y = []

    #setpoints
    sp1_conversion = np.ones(601)
    sp1_conversion[0:100] = 0.9
    sp1_conversion[100:200] = 0.915
    sp1_conversion[200:] = 0.89

    sp2_qc = np.ones(601)

    pv1_conversion = np.zeros(601)
    pv2_qc = np.ones(601)

    ie1 = 0
    ie2 = 0

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
        KC = KP1
        KD = KD1
        # ubias for controller (initial heater)
        op0 = 0
        # upper and lower bounds on heater level
        ophi = 1
        oplo = 0
        # calculate the error
        error = sp1-pv1
        # calculate the integral error
        ierr += abs(error) * dt
        # calculate the measurement derivative
        dpv = (pv1 - pv1_last) / dt
        # calculate the PID output
        P = KC * error
        I = ierr
        D = -KD * dpv
        # print(f"PROPORTIONAL TERM => {P}")
        op = op0 + P + D
        # implement anti-reset windup
        # if np.all(op < oplo):
        #     I = I - KI * error * dt
        #     # op = max(oplo,min(ophi,op))
        # # return the controller output and PID terms
        # if np.all(op > ophi):
        #     I = I - KI * error * dt
        op = max(oplo,min(ophi,op))
        # print(f"SETPOINT VALUE => {op}")
        # return the controller output and PID terms
        return [op, ierr]

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
        KI = 0
        KD = KD2
        # ubias for controller (initial heater)
        op0 = 0
        # upper and lower bounds on heater level
        ophi = 1
        oplo = 0
        # calculate the error
        error = sp2-pv2
        # calculate the integral error
        ierr2 += abs(error) * dt
        # calculate the measurement derivative
        dpv = (pv2 - pv_last2) / dt
        # calculate the PID output
        P = KP * error
        D = -KD * dpv
        op = op0 + P + D
        # implement anti-reset windup
        # if op < oplo:
        #     I = I - KI * error * dt
        # if op > ophi:
        #     I = I - KI * error * dt
        op = max(oplo,min(ophi,op))
        # return the controller output and PID terms
        return [op, ierr2]

    stability_error_1 = 0.0
    stability_error_2 = 0.0
    for i in range(1,600):
        # ts = [t[i], t[i + 1]]
        if i < 1:
            sp2_qc[i], ie1 = outer_pid(sp1_conversion[i], pv1_conversion[i], 0, ie1, dt)
            U[i], ie2 = inner_pid(sp2_qc[i], pv2_qc[i], 0, ie2, dt)
        else:
            sp2_qc[i], ie1 = outer_pid(sp1_conversion[i], pv1_conversion[i], pv1_conversion[i-1], ie1, dt)
            U[i], ie2 = inner_pid(sp2_qc[i], pv2_qc[i], pv2_qc[i - 1], ie2, dt)

        stability_error_1 += ie1
        stability_error_2 += ie2

        pv2_qc[i+1] = U[i]
        y = odeint(model, yo, [0, dt], args=(U[i], q), tfirst=True)
        yo = y[-1] + np.random.normal(0, 0, 4)
        Ca[i], Cc[i], T[i], Tc[i] = yo[0], yo[1], yo[2], yo[3]
        x = conversion(Ca[i])
        pv1_conversion[i+1] = x

    return (KP2, abs(ie2))


min_ier2 = (0,100)
# stable_kc2 = []

for kc in kc2_test_range:
    k,err = stability_analysis(kc)
    if err < min_ier2[1]:
        min_ier2 = (k,err)
    kc2_stab.append((k,err))

kc, ier = zip(*kc2_stab)

print(min_ier2)

plt.scatter(kc,ier)
# plt.plot(kc, np.ones(len(kc2_test_range))*2, "-b", label="acceptable error")
plt.xlabel("Kc Value")
plt.ylabel("Absolute Value of Error")
plt.show()

