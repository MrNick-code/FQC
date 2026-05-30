import matplotlib.pyplot as plt
import numpy as np
from tdse import get_H_full_n, TDSE_solve
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("fork")

H_func = get_H_full_n(fix_t=False)

def td_charged(args, psi0, ind0, ind1, dt, t_max):
    H_matrix = H_func(*args)
    t_eval, psi = TDSE_solve(
        dt,
        t_max,
        lambda t: H_matrix,
        psi0,
        args=(),
        method="vode"
    )
    return t_eval, psi

# figure
width = 5.90666
colors = [plt.cm.tab10(i) for i in range(10)]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(width*2, width*2 / 1.618))

### parameters ###
values_dict_mi = dict(t=0, alpha=np.pi, mu_M=.45, B=.05, U=0)
values_dict_hop = dict(t=.15, alpha=np.pi/8, mu_M=.45, B=.05, U=0)
values_dict_pair = dict(t=.15, alpha=np.pi/8, mu_M=0, B=.05, U=0)
values_dict_coul = dict(t=0, alpha=np.pi, mu_M=.45, B=.05, U=1)


"""
Comentários:
--> Energy Shift
--> Hopping
--> Superconducting Pairing
--> Coulomb Interaction
- Evidentemente, alpha (theta no artigo) não muda nada pois é relevante apenas para o tunelamento
- U = 0 claramente mantém os estados PSI+ inalterados, felizmente (população = 2 constantemente)
- Quanto maior U, maior a frequência que a população varia no tempo (menor o período): U.A = E.A = hbar.freq ?
- A frequencia de oscilação dos estados PSI+ e PSI- é linear com U (para valores razoáveis em unidades de Delta)
- Não parece ter comportamento com B.
"""

# U1
t_max = 8 * np.pi
dt = 0.01

ind0 = 0 # psi 00
ind1 = 2 # psi 10

H0 = H_func(0, 0, 0, 0, 0)
psi0 = np.zeros(len(H0))

psi0[ind0] = 1/np.sqrt(2)
psi0[ind1] = 1/np.sqrt(2)
t_evals, psi = td_charged(values_dict_mi.values(), psi0, ind0, ind1, dt=dt, t_max=t_max)

psi_plus = (psi[:, ind0] + psi[:, ind1]) / np.sqrt(2)  # A.[PSI_00 + PSI_10]
psi_minus = (psi[:, ind0] - psi[:, ind1]) / np.sqrt(2) # A.[PSI_00 - PSI_10]

ax1.plot(t_evals, np.abs(psi_plus)**2, label=r"$\left|\psi_{L+}\right|^2$", c=colors[6])
ax1.plot(t_evals, np.abs(psi_minus)**2, label=r"$\left|\psi_{L-}\right|^2$", ls='--', c=colors[7])

ax1.axhline(y=.5, ls="--", color="black", alpha=0.3)
ax1.set_xlabel(r"$\hbar \tau /\Delta $")
ax1.set_ylabel("Population")

ax1.set_xlim([0, t_max])
ax1.set_ylim([0, 1.1])
ax1.legend(loc=1, frameon=False)

# U2
t_max = 600
dt = 0.01

ind0 = 2
ind1 = 3

H0 = H_func(0, 0, 0, 0, 0)
psi0 = np.zeros(len(H0))

psi0[ind0] = 1
t_evals, psi = td_charged(values_dict_hop.values(), psi0, ind0, ind1, dt=dt, t_max=t_max)

ax2.plot(t_evals, np.abs(psi[:, ind0]**2), label=r"$\left|\psi_{10}\right|^2$", c=colors[2])
ax2.plot(t_evals, np.abs(psi[:, ind1]**2), label=r"$\left|\psi_{01}\right|^2$", ls='--', c=colors[3])

ax2.axhline(y=.5, ls="--", color="black", alpha=0.3)
ax2.set_xlabel(r"$\hbar \tau /\Delta $")
ax2.set_ylabel("Population")

ax2.set_xlim([0, t_max])
ax2.set_ylim([0, 1.1])
ax2.legend(loc=1, frameon=False)

# U3
t_max = 600
dt = 0.01
ind0 = 0
ind1 = 1

H0 = H_func(0, 0, 0, 0, 0)
psi0 = np.zeros(len(H0))

psi0[ind0] = 1
t_evals, psi = td_charged(values_dict_pair.values(), psi0, ind0, ind1, dt=dt, t_max=t_max)

ax3.plot(t_evals, np.abs(psi[:, ind0]**2), label=r"$\left|\psi_{00}\right|^2$", c=colors[0])
ax3.plot(t_evals, np.abs(psi[:, ind1]**2), label=r"$\left|\psi_{11}\right|^2$", ls='--', c=colors[1])

ax3.axhline(y=.5, ls="--", color="black", alpha=0.3)
ax3.set_xlabel(r"$\hbar \tau /\Delta $")
ax3.set_ylabel("Population")

ax3.set_xlim([0, t_max])
ax3.set_ylim([0, 1.1])
ax3.legend(loc=1, frameon=False)

# U4
t_max_c = 4 * np.pi
dt = 0.01
ind0 = 0
ind1 = 1

H0 = H_func(0, 0, 0, 0, 0)
psi0 = np.zeros(len(H0))

psi0[ind0] = 1/np.sqrt(2)
psi0[ind1] = 1/np.sqrt(2)

t_evals, psi = td_charged(values_dict_coul.values(), psi0, ind0, ind1, dt=dt, t_max=t_max_c)

psi_plus = (psi[:, ind0] + psi[:, ind1]) / np.sqrt(2)
psi_minus = (psi[:, ind0] - psi[:, ind1]) / np.sqrt(2)

ax4.plot(t_evals, np.abs(psi_plus)**2, label=r"$\left|\psi_{+}\right|^2$", c=colors[4])
ax4.plot(t_evals, np.abs(psi_minus)**2, ls='--', label=r"$\left|\psi_{-}\right|^2$", c=colors[5])

ax4.axhline(y=.5, ls="--", color="black", alpha=0.3)
ax4.set_xlabel(r"$\hbar \tau /\Delta $")
ax4.set_ylabel("Population")

ax4.set_xlim([0, t_max_c])
ax4.set_ylim([0, 1.1])

ax4.legend(loc=1, frameon=False)

"""
# f(U)

# U_log = np.logspace(-34, -2, 75)
# U_lin = np.linspace(-2, 1, 50) 
# U_values = np.concatenate([U_log, 10**U_lin])

t_max_c2 = t_max_c * 10

U_values = np.linspace(0, 500, 150)
frequencies = []

for U in U_values:
    values_dict_ = dict(t=0, alpha=np.pi/6, mu_M=.45, B=0.05, U=U)
    print(U)

    t_evals_, psi_ = td_charged(values_dict_.values(), psi0, ind0, ind1, dt=dt, t_max=t_max_c)

    psi_plus_ = (psi_[:, ind0] + psi_[:, ind1]) / np.sqrt(2)
    signal = np.abs(psi_plus_)**2

    signal = signal - np.mean(signal)

    # FFT
    N = len(signal)
    dt_local = t_evals_[1] - t_evals_[0]

    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=dt_local)

    mask = freqs > 0
    freqs = freqs[mask]
    fft_vals = np.abs(fft_vals[mask])

    freq_dom = freqs[np.argmax(fft_vals)]

    frequencies.append(freq_dom)

ax5.plot(U_values, frequencies, marker='.')

ax5.set_xlabel("U")
ax5.set_ylabel(r"$\omega_{U_m}(\left|\psi_{+}\right|^2) = \left(\dfrac{\hbar \tau^{-1}}{\Delta}\right)$")

# ax5.set_xscale("log") 
ax5.grid(alpha=0.3)
"""

plt.tight_layout()
plt.show()

# -------------------------------------------------------


