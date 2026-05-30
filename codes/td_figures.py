# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tdse import TD_pair_pulse, TD_hop_pulse, sigmoid_pulse, H_full_n, TDSE_solve, TD_coulomb_pulse
from matplotlib.gridspec import GridSpec

tex_settings = {
    "text.usetex": False,                                                                                # MATHEUS CAPELIN changed True to False argument
    "font.family": "sans-serif",
    "font.serif": "Computer Modern Roman",
    "font.size": 10,
    "text.latex.preamble": r"\usepackage{siunitx} \usepackage[version=4]{mhchem} \sisetup{detect-all}",
}
mpl.rcParams.update(tex_settings)

# %%
# all for t=0.3 alpha=np.pi/8, B = 0.2
values_dict_hop = {
    "transition": {"pulse_correction": 1.05, "rise_time": 0.5, "method": "dopri5"},
    "wiggly": {"pulse_correction": 1.08, "rise_time": 2},
    "smooth": {"pulse_correction": 1.28, "rise_time": 6},
}

values_dict = {
    "transition": {"pulse_correction": 1.03, "rise_time": 0.5, "method": "dopri5"},
    "wiggly": {"pulse_correction": 1.12, "rise_time": 2},
    "smooth": {"pulse_correction": 1.41, "rise_time": 6},
}

values_dict_coul = {
    "transition": {"pulse_correction": 1.0, "rise_time": 0.5, "method": "dopri5"},
    "wiggly": {"pulse_correction": 1.15, "rise_time": 2},
    "smooth": {"pulse_correction": 1.30, "rise_time": 6}
}

# %%
times = []
psis = []
pulse_params = []

for key in values_dict:
    time, psi, pulse_param = TD_pair_pulse(**values_dict[key], alpha=np.pi / 8, B=0.2)
    pulse_params.append(pulse_param)
    times.append(time)
    psis.append(psi)

times_hop = []
psis_hop = []
pulse_params_hop = []

for key in values_dict_hop:
    time, psi, pulse_param = TD_hop_pulse(**values_dict_hop[key], alpha=np.pi / 8, mu_M=0.45, B=0.2)
    pulse_params_hop.append(pulse_param)
    times_hop.append(time)
    psis_hop.append(psi)

times_coul = []
psis_coul = []
pulse_params_coul = []

for key in values_dict_coul:
    time, psi, pulse_param = TD_coulomb_pulse(**values_dict_coul[key], alpha=np.pi/1, mu_M=0.45, B=0.2)


# %%

def TD_plot(
        times, 
        psis, 
        pulse_params, 
        initial_state=(2, r"$\left|\psi_{10}\right|^2$"),
        final_state=(3, r"$\left|\psi_{01}\right|^2$"),
        filename="../publication/figures/TD_hop_gate.pdf",
        abs_ylim=(0, 0.27)
        ):
    t_start = 15
    xlim = 40
    colors = [plt.cm.tab10(i) for i in range(3)]

    width = 5.90666
    pulse_amp = 0.3

    fig = plt.figure(figsize=(width, width / 1.618))
    ax_dict = fig.subplot_mosaic(
        [
            ["A", "B"],
            ["C", "B"],
        ],
    )

    ax2 = ax_dict['C']
    ax0 = ax_dict['A']
    ax1 = ax_dict['B']

    ax0.text(0.4, 0.3, "a)", verticalalignment="top")
    ax1.text(0.4, 0.985, "b)", verticalalignment="top")
    ax2.text(0.4, abs_ylim[1], "c)", verticalalignment="top")


    ax0.set_ylabel(r"$t/\Delta$")
    ax1.set_ylabel(r"Population")
    ax2.set_ylabel("ABS population")

    ax0.set_ylim([0, pulse_amp + 0.01])
    ax1.set_ylim([0, 1.03])
    ax2.set_ylim(abs_ylim)

    ax0.set_xlim([0, xlim])
    ax1.set_xlim([0, xlim])
    ax2.set_xlim([0, xlim])

    ax1.set_yticks([0, 0.5, 1])
    ax1.axhline(y=0.5, ls="--", color="black", alpha=0.3)

    #ax0.set_xlabel(r"$\hbar \tau /\Delta $")
    ax0.set_xticklabels([])
    ax1.set_xlabel(r"$\hbar \tau /\Delta $")
    ax2.set_xlabel(r"$\hbar \tau /\Delta $")

    
    for i in range(len(psis)):
        time = times[i] - t_start
        psi = psis[i]
        pulse_param = pulse_params[i]

        rise_time = pulse_param[1]
        population_0 = np.abs(psi[:, initial_state[0]]) ** 2
        population_2 = np.abs(psi[:, final_state[0]]) ** 2
        ABS_pop = np.sum(np.abs(psi[:, 4:]) ** 2, axis=1)
        pulse = sigmoid_pulse(time + t_start, *pulse_param) * pulse_amp
        
        ax0.plot(time, pulse, color=colors[i], label=f"{rise_time}")

        if i==0:
            ax1.plot(
                time,
                population_0,
                color=colors[i],
                ls="--",
                label=initial_state[1],
            )
            ax1.plot(
                time,
                population_2,
                color=colors[i],
                label=final_state[1],
            )
        else:
            ax1.plot(
                time,
                population_0,
                color=colors[i],
                ls="--",
            )
            ax1.plot(
                time,
                population_2,
                color=colors[i],
            )
        ax2.plot(time, ABS_pop, color=colors[i], label=f"{rise_time}")

    ax2.legend(loc=1, frameon=False, title=r"$\hbar \tau_R/ \Delta =$", bbox_to_anchor=(1.05, 1))
    ax1.legend(loc=1, frameon=False)
    plt.tight_layout()
    #plt.savefig(
    #    filename,
    #    bbox_inches="tight",
    #    transparent=False,
    #    dpi=500,
    #)
    plt.show()

    
# %%
TD_plot(times_hop, psis_hop, pulse_params_hop, abs_ylim=(0, 0.45))

# %%
TD_plot(
    times,
    psis,
    pulse_params,
    filename="../publication/figures/TD_pair_gate.pdf",
    initial_state=(0, r"$\left|\psi_{00}\right|^2$"),
    final_state=(1, r"$\left|\psi_{11}\right|^2$"),
)

# %%
TD_plot(
    times_coul,
    psis_coul,
    pulse_params_coul,
    initial_state=(0, r"$\left|\psi_{00}\right|^2$"),
    final_state=(1, r"$\left|\psi_{11}\right|^2$")
)


# %%
def td_charged(args, psi0, ind0, ind1, dt=0.1, t_max=1000):
    _H_full_n = H_full_n(*args)
    t_eval, psi = TDSE_solve(dt, t_max, lambda t: _H_full_n, psi0, args=(), method="vode")   
    return t_eval, psi
    
values_dict = dict(t=0.15, alpha=np.pi/8, mu_M = 0, B = 0.05)
t_max= 600
t_max_c = 4*np.pi * 10 # times 10 to see the wave group 
dt = 0.01
width = 5.90666
colors = [plt.cm.tab10(i) for i in range(4)]
fig = plt.figure(figsize=(width*2, width / 1.618)) # Matheus: add *2
(ax1, ax2, ax3) = fig.subplots(1, 3)

# Pairing
ind0 = 0
ind1 = 1

psi0 = np.zeros(len(H_full_n(0, 0, 0, 0)))
psi0[ind0] = 1
t_evals, psi = td_charged(values_dict.values(), psi0, ind0, ind1, dt=dt, t_max=t_max)

ax1.plot(t_evals, np.abs(psi[:, ind0]**2), label=r"$\left|\psi_{00}\right|^2$", c=colors[0])
ax1.plot(t_evals, np.abs(psi[:, ind1]**2), label=r"$\left|\psi_{11}\right|^2$", ls='--', c=colors[1])


# Hopping
ind0 = 2
ind1 = 3

psi0 = np.zeros(len(H_full_n(0, 0, 0, 0)))
psi0[ind0] = 1
values_dict['mu_M'] = 0.45
t_evals, psi = td_charged(values_dict.values(), psi0, ind0, ind1, dt=dt, t_max=t_max)

ax2.plot(t_evals, np.abs(psi[:, ind0]**2), label=r"$\left|\psi_{10}\right|^2$", c=colors[2])
ax2.plot(t_evals, np.abs(psi[:, ind1]**2), label=r"$\left|\psi_{01}\right|^2$", ls='--', c=colors[3])


# Coulomb Interaction
ind0 = 0
ind1 = 1

psi0 = np.zeros(len(H_full_n(0, 0, 0, 0)))
psi0[0] = 1/np.sqrt(2)
psi0[1] = 1/np.sqrt(2)
values_dict['mu_M'] = 0.45
t_evals, psi = td_charged(values_dict.values(), psi0, ind0, ind1, dt=dt, t_max=t_max_c)

psi_plus = (psi[:, ind0] + psi[:, ind1]) / np.sqrt(2) # Matheus: psi + and - states
psi_minus = (psi[:, ind0] - psi[:, ind1]) / np.sqrt(2)

ax3.plot(t_evals, np.abs(psi_plus)**2, label=r"$\left|\psi_{+}\right|^2$", c=colors[2])
ax3.plot(t_evals, np.abs(psi_minus)**2, label=r"$\left|\psi_{-}\right|^2$", ls='--', c=colors[3])


# figurre set up 
ax1.set_ylabel(r"Population")
ax1.set_yticks([0, 0.5, 1])
ax1.axhline(y=0.5, ls="--", color="black", alpha=0.3)
ax1.set_xlabel(r"$\hbar \tau /\Delta $")
ax2.axhline(y=0.5, ls="--", color="black", alpha=0.3)
ax2.set_xlabel(r"$\hbar \tau /\Delta $")
ax3.axhline(y=0.5, ls="--", color="black", alpha=0.3)
ax3.set_xlabel(r"$\hbar \tau /\Delta $")

ax1.text(0.02, 0.99, "a)", horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)
ax2.text(0.02, 0.99, "b)", horizontalalignment='left', verticalalignment='top', transform=ax2.transAxes)
ax3.text(0.02, 0.99, "c)", horizontalalignment='left', verticalalignment='top', transform=ax3.transAxes)

ax1.set_xlim([0, t_max])
ax1.set_ylim([0, 1.1])

ax2.set_xlim([0, t_max])
ax2.set_ylim([0, 1.1])

ax3.set_xlim([0, t_max_c])
ax3.set_ylim([0, 1.1])

ax2.set_yticklabels([])
ax2.set_yticks([0, 0.5, 1])

ax1.legend(loc=1, frameon=False, bbox_to_anchor=(1, 1), ncol=3)
ax2.legend(loc=1, frameon=False, bbox_to_anchor=(1, 1), ncol=3)
ax3.legend(loc=1, frameon=False, bbox_to_anchor=(1, 1), ncol=3)

plt.tight_layout()
#plt.savefig(
#    "../publication/figures/TD_constant_charged.pdf",
#    bbox_inches="tight",
#    transparent=False,
#    dpi=500,
#)
plt.show()

# %%
