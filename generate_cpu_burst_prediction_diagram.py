import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

process_bursts = np.array([6, 4, 6, 4, 13, 13, 13, 13, 13])
tau = [10]

alpha: np.double = .5

for process_time in process_bursts:
    tau.append(alpha * process_time + (1 - alpha) * tau[-1])

sns.set_style('darkgrid')
# make background white

plt.grid(True)

MARGIN: np.double = 0.2

plt.figure(figsize=(10, 5))

plt.xticks(range(len(process_bursts)))
plt.yticks(range(2, int(np.max(process_bursts)), 2))

plt.xlim(0, len(process_bursts) - 2 * MARGIN)
plt.ylim(0, np.max(process_bursts) + 2 * MARGIN)

plt.step(range(len(process_bursts)), process_bursts, color='black', linewidth=1.7, where='post', label='CPU-burst - $x_t$')

x_tau = np.linspace(0, len(tau) - 1.6, 1_000)

spl = make_interp_spline(range(len(tau)), tau, k=2)
tau_smooth = spl(x_tau)

plt.plot(x_tau, tau_smooth, color='b', linewidth=1.7, label='Prediction - $\\tau_t$')

# line for Ox
plt.axhline(0, color='black', linewidth=4)
plt.axvline(0, color='black', linewidth=4)

# add top and right lines
plt.axhline(np.max(process_bursts) + 2 * MARGIN, color='black', linewidth=2.7)
plt.axvline(len(process_bursts) - 2 * MARGIN, color='black', linewidth=2.7)

plt.xlabel('Eșantion')
plt.ylabel('Timp [ms]')
plt.legend()

plt.title('Predicția timpului noului CPU-burst')
plt.tight_layout()

plt.savefig('cpu_burst_prediction.pdf')

# plt.show()