import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters for the Bannister Impulse Response Model
k1 = 0.26  # Fitness scaling factor
k2 = k1*1.5  # Fatigue scaling factor
tau1 = 28  # Fitness decay constant (days)
tau2 = 10  # Fatigue decay constant (days)
p0 = 180  # Starting FTP (W) 160/62=2.58

# Example TSS data
#tss = np.array([0, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75])  # Simulated TSS values
data = pd.read_csv('TSS - Sheet1.csv')
tss = data['TSS']
days = np.arange(len(tss))  # Adjusted to match the length of tss

# Initialize fitness, fatigue, and predicted FTP
PTE = np.zeros(len(days))
NTE = np.zeros(len(days))
performance = np.zeros(len(days))

# Starting fitness and fatigue values
PTE[0] = 20  # Initial fitness value 
NTE[0] = 0    # Initial fatigue value

# Calculate fitness, fatigue, and predicted FTP
for t in range(1, len(days)):
    # Exponential decay of fitness and fatigue
    PTE[t] = PTE[t-1] * np.exp(-1/tau1) + tss[t] * k1
    NTE[t] = NTE[t-1] * np.exp(-1/tau2) + tss[t] * k2
    # Performance calculation
    performance[t] = p0 + (PTE[t]) - (NTE[t]) 

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('time (days)')
ax1.set_ylabel('Predicted FTP (W)')
l1, = ax1.plot(days, performance, label='Predicted FTP (W)', color='blue')
plt.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('AU')
l2, = ax2.plot(days, PTE, label='PTE', color='red')
l3, = ax2.plot(days, NTE, label='NTE', color='green')

plt.title('Bannister Impulse Response Model - FTP Prediction')
plt.legend([l1, l2, l3], ['Predicted FTP (W)', 'PTE', 'NTE'])
plt.show()

print("FTP_peak = "+str(max(performance)))

