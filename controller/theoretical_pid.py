import time
import tclab
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# initial parameters
Kp1 = 0.9
Kd = 0.5
tau_p = 148.8
theta_p = 15.3


def heat(x,t,Q1,Q2):
    # Optimized parameters
    # U,alpha1,alpha2 = p
    U = 9.5474966446
    Us = 37.396276968
    alpha1 = 0.014142850127
    alpha2 = 0.0061013478688



    # Parameters
    Ta = 23 + 273.15   # K
    m = 4.0/1000.0     # kg
    Cp = 0.5 * 1000.0  # J/kg-K
    A = 10.0 / 100.0**2 # Area in m^2
    As = 2.0 / 100.0**2 # Area in m^2
    eps = 0.9          # Emissivity
    sigma = 5.67e-8    # Stefan-Boltzman

    # Temperature States
    T1 = x[0] + 273.15
    T2 = x[1] + 273.15

    # Heat Transfer Exchange Between 1 and 2
    conv12 = U*As*(T2-T1)
    rad12  = eps*sigma*As * (T2**4 - T1**4)

    # Nonlinear Energy Balances
    dT1dt = (1.0/(m*Cp))*(U*A*(Ta-T1) \
            + eps * sigma * A * (Ta**4 - T1**4) \
            + conv12 + rad12 \
            + alpha1*Q1)
    dT2dt = (1.0/(m*Cp))*(U*A*(Ta-T2) \
            + eps * sigma * A * (Ta**4 - T2**4) \
            - conv12 - rad12 \
            + alpha2*Q2)

    return [dT1dt,dT2dt]


# -----------------------------
# Input Kc,tauI,tauD
# -----------------------------
Kp1 = 0.9
Kd = 0.5
tau_p = 148.8
theta_p = 15.3

Kp2 = 0.5 * Kp1
tau_c = tau_p

tau_i = tau_p + 0.5 * theta_p
tau_d = (tau_p * theta_p)/(2 * tau_p + theta_p)


def pid2(sp,pv,pv_last,ierr,dt, heater):
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

n = 600  # Number of second time points (10 min)
T0 = 296.15 - 273.15
tm = np.linspace(0,n,n+1) # Time values
T1 = np.zeros(n+1)
T2 = np.zeros(n+1)
T1[0] = T0
T2[0] = T0
Q1 = np.zeros(n+1)
Q2 = np.zeros(n+1)
Tsp1 = np.ones(n+1) * T0
Tsp2 = np.ones(n+1) * T0 # set point (degC)
Tsp1[10:] = 60.0       # step up
Tsp1[400:] = 40.0      # step for i in range(n+1):
Tsp2[150:] = 50.0        # step up
Tsp2[350:] = 35.0

ierr1 = 0.0
ierr2 = 0.0
error1 = []
error2 = []

for i in range(1,n):
    # if i > 3:
    #     break
    print(f"{5 * '-'} ITERATION {i} {5 * '-'}")
    delta_t = tm[1] - tm[0]
    ndelay = int(np.ceil(theta_p/delta_t))
    iop = max(0, i - ndelay)
    print(f"IOP => {iop}")
    Q1[i], P, ierr1, D = pid2(Tsp1[i], T1[iop], T1[iop-1], ierr1, delta_t, heater=1)
    Q2[i], P, ierr2, D = pid2(Tsp2[i], T2[iop], T2[iop-1], ierr2, delta_t, heater=2)
    print(f"CURRENT TEMP => {T1[iop]}")
    print(f"PREVIOUS TEMP => {T1[iop-1]}")
    print(f"IERR1 => {ierr1}")
    print(f"IERR2 => {ierr2}")
    error1.append(ierr1)
    error2.append(ierr2)
    # Q1[i] = 50
    # Q2[i] = 0

    ts = [tm[i-1],tm[i]]
    if i == 0:
        T0 = [T1[0], T2[0]]
    else:
        T0 = [T1[i-1], T2[i-1]]
    print(f"HEATER OUTPUT => {Q1[i]}, {Q2[i]}")
    print(f"CHANGE IN TEMPERATURE => {heat(T0, [delta_t], Q1[i], Q2[i])}")
    y = odeint(heat,T0,[0, delta_t],args=(Q1[i],Q2[i]))
    print(f"OUTPUT OF ODEINT => {y[-1]}")
    new_T1 = y[-1][0]
    new_T2 = y[-1][1]
    T1[i] = np.copy(new_T1)
    T2[i] = np.copy(new_T2)

def total_error(T_measured1, Tsp1, T_measured2, Tsp2):
    error = 0.0

    for i in range(len(T_measured1)-1):
        error += ((Tsp1[i] - T_measured1[i])/T_measured1[i]) ** 2 + ((Tsp2[i] - T_measured2[i])/T_measured2[i]) ** 2

    return error


sse = total_error(T1, Tsp1, T2, Tsp2)
print(f"TOTAL ERROR => {sse}")

plt.figure(1,figsize=(15,7))
plt.subplot(2,1,1)
plt.plot(tm,T1,'r.',linewidth=2,label='Temperature 1 (meas)')
plt.plot(tm,T2,'b.',linewidth=2,label='Temperature 2 (meas)')
plt.plot(tm, Tsp1, 'r:', label=r'$Q_1$ SP')
plt.plot(tm, Tsp2, "b:",label=r'$Q_2$ SP')
plt.ylabel(r'T $(^oC)$')
plt.legend(loc=2)
plt.subplot(2,1,2)
plt.plot(tm,Q1,'r--',linewidth=2,label=r'Heater 1 ($Q_1$)')
plt.plot(tm[:599], error1, "r-", linewidth=1, label=r"Error for $Q_1$")
plt.plot(tm[:599], error2, "b-", linewidth=1, label=r"Error for $Q_2$")
plt.plot(tm,Q2,'b:',linewidth=2,label=r'Heater 2 ($Q_2$)')
plt.ylabel(r"U(t)")
plt.legend(loc='best')
plt.xlabel('time (sec)')
# plt.show()
