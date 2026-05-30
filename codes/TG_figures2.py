import matplotlib.pyplot as plt
import numpy as np
from tdse import TD_pair_pulse, TD_hop_pulse, sigmoid_pulse, TD_coulomb_pulse, TD_phase_pulse
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

if __name__ == "__main__":
    mp.set_start_method("fork")

# Directories
values_dict_hop = {
    "transition": {"pulse_correction": 1.05, "rise_time": 0.5, "method": "dopri5"},
    "wiggly": {"pulse_correction": 1.08, "rise_time": 2, "method": "dopri5"},
    "smooth": {"pulse_correction": 1.28, "rise_time": 6, "method": "dopri5"},
}

values_dict = {
    "transition": {"pulse_correction": 1.03, "rise_time": 0.5, "method": "dopri5"},
    "wiggly": {"pulse_correction": 1.12, "rise_time": 2, "method": "dopri5"},
    "smooth": {"pulse_correction": 1.41, "rise_time": 6, "method": "dopri5"},
}

values_dict_phase = {
    "transition": {"pulse_correction": 1.05, "rise_time": 0.5, "method": "dopri5"},
    "wiggly": {"pulse_correction": 1.08, "rise_time": 2, "method": "dopri5"},
    "smooth": {"pulse_correction": 1.28, "rise_time": 6, "method": "dopri5"}
}

values_dict_coul = {
    "transition": {"pulse_correction": 1.05, "rise_time": 0.5, "method": "dopri5"},
    "wiggly": {"pulse_correction": 1.08, "rise_time": 2, "method": "dopri5"},
    "smooth": {"pulse_correction": 1.28, "rise_time": 6, "method": "dopri5"}
}
'''
print("1..//4")
times = []
psis = []
pulse_params = []

def run_case(key):
    return TD_pair_pulse(**values_dict[key], alpha=np.pi / 8, B=.2, U=0)

with ThreadPool() as pool:
    results = pool.map(run_case, values_dict.keys())

for time, psi, pulse_param in results:
    times.append(time)
    psis.append(psi)
    pulse_params.append(pulse_param)


print("2..//4")
times_hop = []
psis_hop = []
pulse_params_hop = []

def run_case2(key):
    return TD_hop_pulse(**values_dict_hop[key], alpha=np.pi / 8, mu_M=.45, B=.2, U=0)

with ThreadPool() as pool:

    results_hop = pool.map(run_case2, values_dict_hop.keys())

for time, psi, pulse_param in results_hop:
    times_hop.append(time)
    psis_hop.append(psi)
    pulse_params_hop.append(pulse_param)


print("3..//4")
times_coul = []
psis_coul = []
pulse_params_coul = []

def run_case3(key):
    return TD_coulomb_pulse(**values_dict_coul[key], alpha=np.pi, mu_M=0, B=.2, U=1)

with ThreadPool() as pool:

    results_coul = pool.map(run_case3, values_dict_coul.keys())

for time, psi, pulse_param in results_coul:
    times_coul.append(time)
    psis_coul.append(psi)
    pulse_params_coul.append(pulse_param)


print("3.5..//4")
times_phase = []
psis_phase = []
pulse_params_phase = []

def run_case4(key):
    return TD_phase_pulse(**values_dict_phase[key], alpha=np.pi, mu_M=1, B=.2, U=0)

with ThreadPool() as pool:

    results_phase = pool.map(run_case4, values_dict_phase.keys())

for time, psi, pulse_param in results_phase:
    times_phase.append(time)
    psis_phase.append(psi)
    pulse_params_phase.append(pulse_param)
'''
    
print("plot..//4")

# Plot function
def TD_plot(
        times, 
        psis, 
        pulse_params, 
        initial_state=(2, r"$\left|\psi_{10}\right|^2$"),
        final_state=(3, r"$\left|\psi_{01}\right|^2$"),
        filename="../publication/figures/TD_hop_gate.pdf",
        abs_ylim=(0, 0.27),
        states_mode="normal",
        traced=0.5
        ):
    
    if states_mode == "normal":
        t_start = 15
        pulse_amp = 0.3
        xlim = 40
    else:
        t_start = 0
        pulse_amp = .3
        xlim = 12.57
        # xlim = max([np.max(t) for t in times])
    colors = [plt.cm.tab10(i) for i in range(3)]

    width = 5.90666

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

    ax0.set_ylabel(r"$t/\Delta$")
    ax1.set_ylabel(r"Population")

    ax0.set_ylim([0, pulse_amp + 0.01])
    ax1.set_ylim([0, 1.03])

    ax0.set_xlim([0, xlim])
    ax1.set_xlim([0, xlim])

    ax1.set_yticks([0, 0.5, 1])
    ax1.axhline(y=traced, ls="--", color="black", alpha=0.3)

    #ax0.set_xlabel(r"$\hbar \tau /\Delta $")
    ax0.set_xticklabels([])
    ax1.set_xlabel(r"$\hbar \tau /\Delta $")

    if states_mode == "normal": 
        ax2.text(0.4, abs_ylim[1], "c)", verticalalignment="top")
        ax2.set_ylabel("ABS population")
        ax2.set_ylim(abs_ylim)
        ax2.set_xlim([0, xlim])
        ax2.set_xlabel(r"$\hbar \tau /\Delta $")
    else:
        ax2.text(.4, abs_ylim[1], "c)", verticalalignment="top")
        ax2.set_ylabel("phase variation")
        ax2.set_ylim([-np.pi-0.3, np.pi+0.3])
        ax2.set_xlim([0, xlim])
        ax2.set_xlabel(r"$\hbar \tau /\Delta $")
        ax2.axhline(y=-np.pi, ls="--", color="black", alpha=0.3)
        ax2.axhline(y=np.pi, ls="--", color="black", alpha=0.3)
    
    for i in range(len(psis)):
        time = times[i] - t_start
        psi = psis[i]
        pulse_param = pulse_params[i]

        rise_time = pulse_param[1]
        if states_mode == "normal":
            population_0 = np.abs(psi[:, initial_state[0]]) ** 2
            population_2 = np.abs(psi[:, final_state[0]]) ** 2

        elif states_mode == "coulomb":
            psi_plus = (psi[:, 0] + psi[:, 1]) / np.sqrt(2)
            psi_minus = (psi[:, 0] - psi[:, 1]) / np.sqrt(2)

            population_0 = np.abs(psi_plus)**2
            population_2 = np.abs(psi_minus)**2
        
        elif states_mode == "phase":
            psi_plus = (psi[:, 0] + psi[:, 2]) / np.sqrt(2)
            psi_minus = (psi[:, 0] - psi[:, 2]) / np.sqrt(2)

            population_0 = np.abs(psi_plus)**2
            population_2 = np.abs(psi_minus)**2

        ABS_pop = np.sum(np.abs(psi[:, 4:]) ** 2, axis=1)
        pulse = sigmoid_pulse(time + t_start, *pulse_param) * pulse_amp

        c0 = psi[:, 0]
        if states_mode == "phase":
            c1 = psi[:, 2]
        else: 
            c1 = psi[:, 1]
        norm = np.sqrt(np.abs(c0)**2 + np.abs(c1)**2)
        c0 = c0 / norm
        c1 = c1 / norm

        phase_pop0 = np.unwrap(np.angle(c1) + np.angle(c0))
        phase_pop1 = np.unwrap(-np.angle(c1) + np.angle(c0))
        
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
        
        if states_mode == "normal":
            ax2.plot(time, ABS_pop, color=colors[i], label=f"{rise_time}", ls="--")
        else: 
            ax2.plot(time, phase_pop0, color=colors[i], label="_nolegend_", ls="--")
            ax2.plot(time, phase_pop1, color=colors[i], label=f"{rise_time}")

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

print("4..//4")
# individual plot
'''
TD_plot(times_hop, psis_hop, pulse_params_hop, abs_ylim=(0, 0.45))

TD_plot(
    times,
    psis,
    pulse_params,
    filename="../publication/figures/TD_pair_gate.pdf",
    initial_state=(0, r"$\left|\psi_{00}\right|^2$"),
    final_state=(1, r"$\left|\psi_{11}\right|^2$"),
)


TD_plot(
    times_coul, 
    psis_coul, 
    pulse_params_coul, 
    filename="../publication/figures/TD_coul_gate.pdf", 
    initial_state=(None, r"$|\psi_{+}|^2$"),
    final_state=(None, r"$|\psi_{-}|^2$"), 
    abs_ylim=(0, 0.5),
    states_mode="coulomb",
    traced=1.0
)


TD_plot(
    times_phase, 
    psis_phase, 
    pulse_params_phase, 
    filename="../publication/figures/TD_phase_gate.pdf",
    initial_state=(None, r"$|\psi_{L+}|^2$"),
    final_state=(None, r"$|\psi_{L-}|^2$"), 
    abs_ylim=(0, 0.5),
    states_mode="phase",
    traced=1.0
)'''
# -------------------------------------
''''''
from tdse import get_H_full_n, TD_general_pulse
H_pair_num = get_H_full_n(fix_t=False)

H_hop_num = get_H_full_n(fix_t=False)

H_coul_num = get_H_full_n(fix_t=True)

H_phase_num = get_H_full_n(fix_t=True)

# multigates - initial state
psi0 = np.zeros(16, dtype=complex)
psi0[0] = 1

def H_pair(pulse, alpha, B, U):
    return H_pair_num(0.3 * pulse,alpha,0,B,U)

t1, psi1 = TD_general_pulse(psi0,H_pair,pulse_args=(np.pi/8, 0.2, 0), 
                            idle_time_aft=5,pulse_duration=16,rise_time=1)
psi_after_1 = psi1[-1]

def H_coul(pulse, alpha, B, U):
    return H_coul_num(0,alpha,0,B,U * pulse)

t2, psi2 = TD_general_pulse(psi_after_1,H_coul,pulse_args=(np.pi/8, 0.2, 1.0),
                            pulse_duration=2,rise_time=1,time_offset=t1[-1],idle_time_aft=5)
psi_after_2 = psi2[-1]

def H_hop(pulse, alpha, mu_M, B):
    return H_hop_num(0.3 * pulse,alpha,mu_M * pulse,B,0)

t3, psi3 = TD_general_pulse(psi_after_2,H_hop,pulse_args=(np.pi/8, 0.45, 0.2),
                            pulse_duration=7,rise_time=1,time_offset=t2[-1],idle_time_aft=5)
psi_after_3 = psi3[-1]

def H_phase(pulse, alpha, mu_M, B, U):
    return H_phase_num(0,alpha,mu_M * pulse,B,U)

t4, psi4 = TD_general_pulse(psi_after_3,H_phase,pulse_args=(np.pi/8, 1.0, 0.2, 0),
                            idle_time_aft=5, pulse_duration=2,rise_time=1,time_offset=t3[-1])

times = np.concatenate([t1, t2, t3, t4])
psis = np.concatenate([psi1,psi2,psi3,psi4])

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(
    times,
    np.abs(psis[:,0])**2,
    label=r"$\left|00\right|^2$",
    lw=1.5
)

ax.plot(
    times,
    np.abs(psis[:,1])**2,
    label=r"$\left|11\right|^2$",
    lw=1.5
)

psi_plus = (
    psis[:,0] + psis[:,1]
)/np.sqrt(2)

psi_minus = (
    psis[:,0] - psis[:,1]
)/np.sqrt(2)

psiR_minus = (
    psis[:,0] - psis[:,3]
)/np.sqrt(2)

psiL_minus = (
    psis[:,0] - psis[:,2]
)/np.sqrt(2)

ax.plot(
    times,
    np.abs(psi_plus)**2,
    ls="--",
    lw=1,
    label=r"$\left|\psi_{+}\right|^2$"
)

ax.plot(
    times,
    np.abs(psi_minus)**2,
    ls="--",
    lw=1,
    label=r"$\left|\psi_{-}\right|^2$"
)

ax.plot(
    times,
    np.abs(psiR_minus)**2,
    ls=":",
    lw=1,
    label=r"$\left|\psi_{R-}\right|^2$"
)

ax.plot(
    times,
    np.abs(psiL_minus)**2,
    ls=":",
    lw=1,
    label=r"$\left|\psi_{L-}\right|^2$"
)

ABS_pop = np.sum(np.abs(psis[:, 4:]) ** 2, axis=1)
ax.plot(
    times, 
    ABS_pop, 
    ls=(5, (10, 3)), 
    lw=2, 
    label="ABS pop."
)

ax.axvline(t1[-1], color="black", alpha=0.2)
ax.axvline(t2[-1], color="black", alpha=0.2)
ax.axvline(t3[-1], color="black", alpha=0.2)

ax.set_xlabel(r"$\hbar\tau/\Delta$")
ax.set_ylabel("Population")

ax.set_ylim([0,1.01])

ax.legend(
    frameon=False,
    ncol=2
)

plt.tight_layout()
plt.show()

# ----------------------------------------
# Other plots tries
def get_even_bloch(psis):
    """
    even parity subspace: |00>, |11>
    """

    c0 = psis[:,0]
    c1 = psis[:,1]

    norm = np.sqrt(np.abs(c0)**2 + np.abs(c1)**2)

    norm[norm == 0] = 1

    c0 = c0 / norm
    c1 = c1 / norm

    X = 2*np.real(np.conj(c0)*c1)
    Y = 2*np.imag(np.conj(c0)*c1)
    Z = np.abs(c0)**2 - np.abs(c1)**2

    return X, Y, Z


def get_odd_bloch(psis):
    """
    odd parity subspace: |10>, |01>
    """

    c0 = psis[:,2]
    c1 = psis[:,3]

    norm = np.sqrt(np.abs(c0)**2 + np.abs(c1)**2)

    norm[norm == 0] = 1

    c0 = c0 / norm
    c1 = c1 / norm

    X = 2*np.real(np.conj(c0)*c1)
    Y = 2*np.imag(np.conj(c0)*c1)
    Z = np.abs(c0)**2 - np.abs(c1)**2

    return X, Y, Z


def get_relative_phase(psis, ind0, ind1):

    c0 = psis[:,ind0]
    c1 = psis[:,ind1]

    norm = np.sqrt(np.abs(c0)**2 + np.abs(c1)**2)

    norm[norm == 0] = 1

    c0 = c0 / norm
    c1 = c1 / norm

    phase = np.unwrap(np.angle(c1) - np.angle(c0))

    return phase


def get_coherence(psis, ind0, ind1):

    c0 = psis[:,ind0]
    c1 = psis[:,ind1]

    coh = c0*np.conj(c1)

    return (np.real(coh), np.imag(coh), np.abs(coh))

# --- phase

fig, ax = plt.subplots(figsize=(12,5))

phase_even = get_relative_phase(psis, 0, 1)

ax.plot(
    times,
    phase_even,
    lw=2,
    label=r"$\phi_{00,11}$"
)

ax.axvline(t1[-1], color="black", alpha=0.2)
ax.axvline(t2[-1], color="black", alpha=0.2)
ax.axvline(t3[-1], color="black", alpha=0.2)

ax.set_xlabel(r"$\hbar\tau/\Delta$")
ax.set_ylabel("Relative phase")

ax.legend(frameon=False)

plt.tight_layout()
plt.show()

# --- coer.
fig, ax = plt.subplots(figsize=(12,5))

coh_re, coh_im, coh_abs = get_coherence(
    psis,
    0,
    1
)

ax.plot(
    times,
    coh_re,
    lw=2,
    label=r"$\mathrm{Re}(\rho_{01})$"
)

ax.plot(
    times,
    coh_im,
    lw=2,
    label=r"$\mathrm{Im}(\rho_{01})$"
)

ax.plot(
    times,
    coh_abs,
    lw=3,
    label=r"$|\rho_{01}|$"
)

ax.axvline(t1[-1], color="black", alpha=0.2)
ax.axvline(t2[-1], color="black", alpha=0.2)
ax.axvline(t3[-1], color="black", alpha=0.2)

ax.set_xlabel(r"$\hbar\tau/\Delta$")
ax.set_ylabel("Coherence")

ax.legend(frameon=False)

plt.tight_layout()
plt.show()

# --- Bloch
X_even, Y_even, Z_even = get_even_bloch(psis)

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(times, X_even, lw=2, label="X")
ax.plot(times, Y_even, lw=2, label="Y")
ax.plot(times, Z_even, lw=2, label="Z")

ax.axvline(t1[-1], color="black", alpha=0.2)
ax.axvline(t2[-1], color="black", alpha=0.2)
ax.axvline(t3[-1], color="black", alpha=0.2)

ax.set_ylim([-1.05,1.05])

ax.set_xlabel(r"$\hbar\tau/\Delta$")
ax.set_ylabel("Bloch coordinates")

ax.legend(frameon=False)

plt.tight_layout()
plt.show()

# --- Bloch path
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------
# EVEN PARITY BLOCH TRAJECTORY
# |00> <-> |11>
# ---------------------------------------------------------

X_even, Y_even, Z_even = get_even_bloch(psis)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# ---------------------------------------------------------
# Bloch sphere surface
# ---------------------------------------------------------

u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)

xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(
    xs,
    ys,
    zs,
    color="gray",
    alpha=0.12,
    linewidth=0
)

# ---------------------------------------------------------
# Great circles
# ---------------------------------------------------------

theta = np.linspace(0, 2*np.pi, 400)

ax.plot(
    np.cos(theta),
    np.sin(theta),
    0*theta,
    alpha=0.15
)

ax.plot(
    np.cos(theta),
    0*theta,
    np.sin(theta),
    alpha=0.15
)

ax.plot(
    0*theta,
    np.cos(theta),
    np.sin(theta),
    alpha=0.15
)

# ---------------------------------------------------------
# Coordinate axes
# ---------------------------------------------------------

ax.plot([-1,1],[0,0],[0,0], lw=1.5)
ax.plot([0,0],[-1,1],[0,0], lw=1.5)
ax.plot([0,0],[0,0],[-1,1], lw=1.5)

# ---------------------------------------------------------
# Axis labels
# ---------------------------------------------------------

ax.text(1.15,0,0,r"$X$", fontsize=16)
ax.text(0,1.15,0,r"$Y$", fontsize=16)

ax.text(
    0,
    0,
    1.18,
    r"$|00\rangle$",
    fontsize=14
)

ax.text(
    0,
    0,
    -1.28,
    r"$|11\rangle$",
    fontsize=14
)

# ---------------------------------------------------------
# Split trajectory by gate
# ---------------------------------------------------------

n1 = len(t1)
n2 = len(t2)
n3 = len(t3)
n4 = len(t4)

# U3
ax.plot(
    X_even[:n1],
    Y_even[:n1],
    Z_even[:n1],
    lw=4,
    label=r"$U_3$"
)

# U4
ax.plot(
    X_even[n1:n1+n2],
    Y_even[n1:n1+n2],
    Z_even[n1:n1+n2],
    lw=4,
    label=r"$U_4$"
)

# U2
ax.plot(
    X_even[n1+n2:n1+n2+n3],
    Y_even[n1+n2:n1+n2+n3],
    Z_even[n1+n2:n1+n2+n3],
    lw=4,
    label=r"$U_2$"
)

# U1
ax.plot(
    X_even[n1+n2+n3:],
    Y_even[n1+n2+n3:],
    Z_even[n1+n2+n3:],
    lw=4,
    label=r"$U_1$"
)

# ---------------------------------------------------------
# Initial/final points
# ---------------------------------------------------------

ax.scatter(
    X_even[0],
    Y_even[0],
    Z_even[0],
    s=120,
    marker="o",
    label="initial"
)

ax.scatter(
    X_even[-1],
    Y_even[-1],
    Z_even[-1],
    s=120,
    marker="*",
    label="final"
)

# ---------------------------------------------------------
# Cosmetics
# ---------------------------------------------------------

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

ax.set_box_aspect([1,1,1])

# remove default axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.grid(False)

# transparent panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# hide cube lines
ax.xaxis.pane.set_edgecolor((1,1,1,0))
ax.yaxis.pane.set_edgecolor((1,1,1,0))
ax.zaxis.pane.set_edgecolor((1,1,1,0))

ax.legend(
    frameon=False,
    loc="upper left"
)

plt.tight_layout()
plt.show()
