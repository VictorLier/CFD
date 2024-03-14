import numpy as np
import matplotlib.pyplot as plt

from FiniteVolume2D import do_simulation

n = 50
L = 1
Pe_ = np.round(np.logspace(1, 7, num=7))
problem = 2
fvscheme = 'uds'

T = np.zeros([len(Pe_), n])
flux = np.zeros([len(Pe_), n])

for i, Pe in enumerate(Pe_):
    _, TT, dT, _, _, _ = do_simulation(n, L, Pe, problem, fvscheme, plot = False)
    T[i, :] = TT[1:-1, 0]
    flux[i, :] = dT[1:-1, 0]


dx = L/n
x = np.linspace(dx/2, L-dx/2, len(TT[1:-1,0]))

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('x')
ax1.set_ylabel('dt', color=color)
ax1.set_ylim([1, 100])
for i in range(len(Pe_)):
    ax1.semilogy(x, flux[i, :], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('TT', color=color)  # we already handled the x-label with ax1
ax2.plot(x, TT[1:-1,0], color=color)
for i in range(len(Pe_)):
    ax2.semilogy(x, T[i, :], color=color)
#ax2.set_ylim([0.01, 7])
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
