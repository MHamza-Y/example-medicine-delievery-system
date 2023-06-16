import numpy as np
from scipy.integrate import odeint

# Define the dynamic equations
def dynamic_system(y, t, k):
    HR, SBP, DBP, T, D = y

    dHRdt = k[0] * D - k[1] * HR
    dSBPdt = k[2] * D - k[3] * SBP
    dDBPdt = k[4] * D - k[5] * DBP
    dTdt = k[6] * D - k[7] * (T - T0)
    dDdt = -k[8] * D

    return [dHRdt, dSBPdt, dDBPdt, dTdt, dDdt]

# Set the initial conditions
HR0 = 75.0  # Initial heart rate
SBP0 = 120.0  # Initial systolic blood pressure
DBP0 = 80.0  # Initial diastolic blood pressure
T0 = 37.0  # Initial temperature
D0 = 0  # Initial dosage

y0 = [HR0, SBP0, DBP0, T0, D0]  # Initial conditions

# Set the parameter values
k = [0.1, 0.05, 0.2, 0.1, 0.15, 0.2, 0.05, 0.1, 0.01]  # Example parameter values

# Set the time points for integration
t = np.linspace(0, 10, 100)  # Time range from 0 to 10 with 100 points

# Solve the dynamic equations
sol = odeint(dynamic_system, y0, t, args=(k,))
# Extract the solution variables
HR = sol[:, 0]
SBP = sol[:, 1]
DBP = sol[:, 2]
T = sol[:, 3]
D = sol[:, 4]

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(t, HR, label='Heart Rate')
plt.plot(t, SBP, label='Systolic BP')
plt.plot(t, DBP, label='Diastolic BP')
plt.plot(t, T, label='Temperature')
plt.plot(t, D, label='Dosage')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()
