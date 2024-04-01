import csv
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def model(time, x, qc, q):
    """
    This function creates the system of differential equations
    needed to model the concentration of reactants and products,
    and the temperature of the reactor and cooling jacket.

    This function serves to be an input for scipy.odeint
    PARAMETERS
    ----------
    t : np.ndarray
        A t span to be used by scipy.odeint to solve the system of differential
        equations
    x : List[Any]
        Inputs to each of the differential equations in the following form:
        x[0] = inputs for dCa/dt - concentration of reactant
        x[1] = inputs for dCc/dt - concentration of product
        x[2] = inputs for dT/dt - reactor
        x[3] = inputs for dTc/dt - cooling jacket
        x[4] = inputs for
    """

    # parameters
    R = 8.14  # J/(mol * k)
    Ea = 50241.6  # Activation Energy, J/mol
    A = 3.19 * 10**8  # pre exponential factor, min^-1
    delta_Hr = -60000  # heat of reaction, J/mol

    V = np.pi / 2  # volume of reactor, m^3
    Cp = 4.186  # heat capacity J/gk
    U = 43500  # overall heat transfer coefficient, J/m^2 min K
    # q = 0.1
    # To = 298.15 # inlet temperature of reactor feed stream, 25 C
    To = 298
    Tco = 273.15  # inlet temperature of coolant stream, 5 C
    rho = 997000  # g/m^3 at SATP
    area = 7.068  # m^2 (side plus bottom)
    Vc = np.pi / 2

    Cao = 500  # mol/m^3
    Cbo = 500  # mol/m^3

    Ca, Cb, Cc, T, Tc = x[0], x[0], x[1], x[2], x[3]

    k = A * np.exp(-Ea / (R * T))
    Ra = -k * Ca * Cb
    Rc = k * Ca * Cb

    dCa_dt = q * (Cao - Ca) + Ra * V
    dCc_dt = -q * Cc + Rc * V
    dT_dt = (
        (q / V) * (To - T)
        + (Ra * delta_Hr) / (rho * Cp)
        - ((U * area) / (rho * V * Cp)) * (T - Tc)
    )
    dTc_dt = (qc / Vc) * (Tco - Tc) + ((U * area) / (rho * Vc * Cp)) * (T - Tc)

    return [dCa_dt, dCc_dt, dT_dt, dTc_dt]


t = np.linspace(0, 600, 601)  # t span, 600 minutes
conversion = [0]

# Step Test
qc = np.ones(601) * 0
qc[200:] = 1

Cao = 100
Cco = 0
To = 298
Tco = To
y0 = [Cao, Cco, To, Tco]

Ca, Cc, T, Tc = np.ones(601), np.ones(601), np.ones(601), np.ones(601)
Ca[0], Cc[0], T[0], Tc[0] = y0[0], y0[1], y0[2], y0[3]

for i in range(len(t) - 1):
    ts = [t[i], t[i + 1]]
    q = random.uniform(0.095, 0.105)  # inlet flow rate to CSTR, m^3/min
    y = odeint(model, y0, ts, args=(qc[i], q), tfirst=True)
    y0 = y[-1] + np.random.normal(0, 0.1, 4)
    Ca[i + 1], Cc[i + 1], T[i + 1], Tc[i + 1] = y0[0], y0[1], y0[2], y0[3]
    x = (Cao - Ca[i + 1]) / (Cao)
    conversion.append(x)

plt.figure()
plt.subplot(2, 1, 1)
# plt.plot(t[100:400], conversion[100:400], label="Conversion of A")
# plt.xlabel("Time (min)")
# plt.ylabel("Conversion (Xa)")
# plt.legend()
plt.plot(t[100:400], Cc[100:400], label="Concentration of C")
plt.xlabel("Time (min)")
plt.ylabel("Concentration of C (mol/m^3)")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t[100:400], qc[100:400], label="Coolant Flow Rate")
plt.xlabel("Time (min)")
plt.ylabel("Coolant Flow Rate (M^3/min)")
plt.legend()
plt.show()

# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(t[:100], T[:100], label="ODEINT T")
# plt.ylabel("Temperature (K)")
# plt.xlabel("Time (min)")
# plt.legend()
# plt.subplot(3, 1, 2)
# plt.ylabel("Concentration (mol/m^3)")
# plt.plot(t[:100], Cc[:100], label="Concentration of C ")
# plt.plot(t[:100], Ca[:100], label="Concentration of A ")
# plt.legend()
# plt.subplot(3, 1, 3)
# plt.plot(t[10:100], conversion[10:100], label="Conversion of A")
# plt.legend()
# plt.show()
