# ============================================================
# Fermionic gates visualization
# Ideal separated parity visualization
# ============================================================

import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("fork")

import matplotlib.pyplot as plt
import numpy as np

from tdse import (
    TD_general_pulse,
    get_H_full_n
)

# ============================================================
# Hamiltonians
# ============================================================

H_pair_num  = get_H_full_n(fix_t=False)
H_hop_num   = get_H_full_n(fix_t=False, fix_lr=True)

H_coul_num  = get_H_full_n(fix_t=True)
H_phase_num = get_H_full_n(fix_t=True, fix_lr=True)

# ============================================================
# PHYSICAL GATES
# ============================================================

# ------------------------------------------------------------
# U3 : EVEN parity Y rotation
# |00> <-> |11>
# ------------------------------------------------------------

def H_U3(pulse, alpha=np.pi/8, B=0.0): # !!!!

    return H_pair_num(
        0.1 * pulse, # !!!!
        alpha,
        0,
        B,
        0
    )

# ------------------------------------------------------------
# U4 : EVEN parity Z rotation
# ------------------------------------------------------------

def H_U4(pulse, alpha=np.pi, B=0.2):

    return H_coul_num(
        0,
        alpha,
        0,
        B,
        1.0 * pulse
    )

# ------------------------------------------------------------
# U2 : ODD parity Y rotation
# |10> <-> |01>
# ------------------------------------------------------------

def H_U2(pulse, alpha=np.pi/8, B=0.2):

    return H_hop_num(
        0.3 * pulse,
        alpha,
        -3 * pulse, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        B,
        0
    )

# ------------------------------------------------------------
# U1 : ODD parity Z rotation
# ------------------------------------------------------------

def H_U1(pulse, alpha=np.pi, B=0.2):

    return H_phase_num(
        0,
        alpha,
        0.45 * pulse,
        B,
        0
    )

# ============================================================
# INITIAL STATES
# ============================================================

# EVEN north pole |00>
psi_even_00 = np.zeros(16, dtype=complex)
psi_even_00[0] = 1

# EVEN equator (+X)
psi_even_x = np.zeros(16, dtype=complex)
psi_even_x[0] = 1/np.sqrt(2)
psi_even_x[1] = 1/np.sqrt(2)

# ODD north pole |10>
psi_odd_10 = np.zeros(16, dtype=complex)
psi_odd_10[2] = 1

# ODD equator (+X)
psi_odd_x = np.zeros(16, dtype=complex)
psi_odd_x[2] = 1/np.sqrt(2)
psi_odd_x[3] = 1/np.sqrt(2)

# ============================================================
# CALIBRATED DURATIONS
# ============================================================

# Longer pulses + smoother ramps
# reduce ABS leakage

T_U3 = 120
T_U2 = 55

T_U4 = 5
T_U1 = 7

RISE_TIME = 6

# ============================================================
# TD RUNNER
# ============================================================

def run_gate(
    H_func,
    psi0,
    pulse_duration,
    rise_time=RISE_TIME,
    idle_time_aft=2
):

    t, psi = TD_general_pulse(
        psi0,
        H_func,
        pulse_args=(),
        pulse_duration=pulse_duration,
        rise_time=rise_time,
        idle_time_aft=idle_time_aft,
        dt=0.01,
        method="dopri5"
    )

    return t, psi

# ============================================================
# BLOCH COORDINATES
# ============================================================

def bloch_coords(psis, i0, i1):

    c0 = psis[:, i0]
    c1 = psis[:, i1]

    norm = np.sqrt(
        np.abs(c0)**2 +
        np.abs(c1)**2
    )

    norm[norm == 0] = 1

    c0 = c0 / norm
    c1 = c1 / norm

    X = 2*np.real(np.conj(c0)*c1)

    Y = 2*np.imag(np.conj(c0)*c1)

    Z = np.abs(c0)**2 - np.abs(c1)**2

    return X, Y, Z

# ============================================================
# BLOCH SPHERE
# ============================================================

def setup_bloch(
    ax,
    title="",
    north=r"$|0\rangle$",
    south=r"$|1\rangle$"
):

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
        alpha=0.08,
        linewidth=0
    )

    theta = np.linspace(0, 2*np.pi, 400)

    # great circles

    ax.plot(
        np.cos(theta),
        np.sin(theta),
        0*theta,
        alpha=0.08
    )

    ax.plot(
        np.cos(theta),
        0*theta,
        np.sin(theta),
        alpha=0.08
    )

    ax.plot(
        0*theta,
        np.cos(theta),
        np.sin(theta),
        alpha=0.08
    )

    # axes

    ax.plot([-1,1],[0,0],[0,0], lw=1.5, alpha=0.5)
    ax.plot([0,0],[-1,1],[0,0], lw=1.5, alpha=0.5)
    ax.plot([0,0],[0,0],[-1,1], lw=1.5, alpha=0.5)

    # labels

    ax.text(1.12,0,0,"X")
    ax.text(0,1.12,0,"Y")

    ax.text(0,0,1.15,north)
    ax.text(0,0,-1.25,south)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.grid(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_box_aspect([1,1,1])

    ax.set_title(
        title,
        pad=5,
        fontsize=13
    )

# ============================================================
# GATE CONFIGURATIONS
# ============================================================

gates = {

    # --------------------------------------------------------
    # U1
    # Odd parity Z rotation
    # --------------------------------------------------------

    "U1": {

        "H": lambda pulse: H_U1(pulse),

        # EVEN parity should remain static
        "psi0_even": psi_even_00,

        # ODD parity starts on equator
        "psi0_odd": psi_odd_x,

        "duration": T_U1
    },

    # --------------------------------------------------------
    # U2
    # Odd parity Y rotation
    # --------------------------------------------------------

    "U2": {

        "H": lambda pulse: H_U2(pulse),

        # EVEN parity static
        "psi0_even": psi_even_00,

        # ODD starts at north pole
        "psi0_odd": psi_odd_10,

        "duration": T_U2
    },

    # --------------------------------------------------------
    # U3
    # Even parity Y rotation
    # --------------------------------------------------------

    "U3": {

        "H": lambda pulse: H_U3(pulse),

        # EVEN starts at north pole
        "psi0_even": psi_even_00,

        # ODD static
        "psi0_odd": psi_odd_10,

        "duration": T_U3
    },

    # --------------------------------------------------------
    # U4
    # Even parity Z rotation
    # --------------------------------------------------------

    "U4": {

        "H": lambda pulse: H_U4(pulse),

        # EVEN equator
        "psi0_even": psi_even_x,

        # ODD static
        "psi0_odd": psi_odd_10,

        "duration": T_U4
    }

}

# ============================================================
# INDIVIDUAL GATES
# ============================================================

for gate_name, gate in gates.items():

    print(f"Running {gate_name}")

    # ========================================================
    # EVEN
    # ========================================================

    t_even, psi_even = run_gate(
        gate["H"],
        gate["psi0_even"],
        gate["duration"]
    )

    X_even, Y_even, Z_even = bloch_coords(
        psi_even,
        0,
        1
    )

    # ========================================================
    # ODD
    # ========================================================

    t_odd, psi_odd = run_gate(
        gate["H"],
        gate["psi0_odd"],
        gate["duration"]
    )

    X_odd, Y_odd, Z_odd = bloch_coords(
        psi_odd,
        2,
        3
    )

    # ========================================================
    # PLOT
    # ========================================================

    fig = plt.figure(figsize=(14,7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    setup_bloch(
        ax1,
        title=f"{gate_name} — EVEN",
        north=r"$|00\rangle$",
        south=r"$|11\rangle$"
    )

    setup_bloch(
        ax2,
        title=f"{gate_name} — ODD",
        north=r"$|10\rangle$",
        south=r"$|01\rangle$"
    )

    # ========================================================
    # EVEN trajectory
    # ========================================================

    ax1.plot(
        X_even,
        Y_even,
        Z_even,
        lw=3
    )

    ax1.scatter(
        X_even[0],
        Y_even[0],
        Z_even[0],
        s=50
    )

    ax1.scatter(
        X_even[-1],
        Y_even[-1],
        Z_even[-1],
        s=120,
        marker="*"
    )

    # ========================================================
    # ODD trajectory
    # ========================================================

    ax2.plot(
        X_odd,
        Y_odd,
        Z_odd,
        lw=3
    )

    ax2.scatter(
        X_odd[0],
        Y_odd[0],
        Z_odd[0],
        s=50
    )

    ax2.scatter(
        X_odd[-1],
        Y_odd[-1],
        Z_odd[-1],
        s=120,
        marker="*"
    )

    plt.tight_layout()
    plt.show()
    
# ============================================================
# SEQUENCE EVOLUTIONS
# ============================================================

# ------------------------------------------------------------
# EVEN sequence
# U3 -> U4
# ------------------------------------------------------------

t1, psi1 = TD_general_pulse(
    psi_even_00,
    lambda pulse: H_U3(0.85 * pulse),
    pulse_args=(),
    pulse_duration=T_U3,
    rise_time=RISE_TIME,
    idle_time_aft=1,
    dt=0.01,
    method="dopri5"
)

psi_after_even = psi1[-1]

t2, psi2 = TD_general_pulse(
    psi_after_even,
    lambda pulse: H_U4(pulse),
    pulse_args=(),
    pulse_duration=T_U4,
    rise_time=RISE_TIME,
    idle_time_aft=1,
    time_offset=t1[-1],
    dt=0.01,
    method="dopri5"
)

times_even = np.concatenate([t1, t2])
psis_even  = np.concatenate([psi1, psi2])

# ============================================================
# ODD sequence
# U2 -> U1
# ============================================================

t3, psi3 = TD_general_pulse(
    psi_odd_10,
    lambda pulse: H_U2(0.72 * pulse),
    pulse_args=(),
    pulse_duration=T_U2,
    rise_time=RISE_TIME,
    idle_time_aft=1,
    dt=0.01,
    method="dopri5"
)

psi_after_odd = psi3[-1]

t4, psi4 = TD_general_pulse(
    psi_after_odd,
    lambda pulse: H_U1(pulse),
    pulse_args=(),
    pulse_duration=T_U1,
    rise_time=RISE_TIME,
    idle_time_aft=1,
    time_offset=t3[-1],
    dt=0.01,
    method="dopri5"
)

times_odd = np.concatenate([t3, t4])
psis_odd  = np.concatenate([psi3, psi4])

# ============================================================
# RELATIVE PHASE
# ============================================================

def relative_phase(psis, i0, i1):

    c0 = psis[:, i0]
    c1 = psis[:, i1]

    norm = np.sqrt(
        np.abs(c0)**2 +
        np.abs(c1)**2
    )

    norm[norm == 0] = 1

    c0 = c0 / norm
    c1 = c1 / norm

    return np.unwrap(
        np.angle(c1) - np.angle(c0)
    )

# ============================================================
# SEQUENCE PLOTS
# ============================================================

def plot_sequence(
    times,
    psis,
    parity="even"
):

    if parity == "even":

        i0, i1 = 0,1

        north = r"$|00\rangle$"
        south = r"$|11\rangle$"

        gate_split = t1[-1]

    else:

        i0, i1 = 2,3

        north = r"$|10\rangle$"
        south = r"$|01\rangle$"

        gate_split = t3[-1]

    X,Y,Z = bloch_coords(psis,i0,i1)

    phase = relative_phase(psis,i0,i1)

    # --------------------------------------------------------
    # Bloch coordinates
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(times,X,lw=2,label="X")
    ax.plot(times,Y,lw=2,label="Y")
    ax.plot(times,Z,lw=2,label="Z")

    ax.axvline(
        gate_split,
        color="black",
        alpha=0.2
    )

    ax.set_ylim([-1.05,1.05])

    ax.set_xlabel(r"$\hbar\tau/\Delta$")

    ax.set_ylabel("Bloch coordinate")

    ax.legend(frameon=False)

    ax.set_title(
        f"{parity.upper()} parity — Bloch coordinates"
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Relative phase
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        times,
        phase,
        lw=2
    )

    ax.axvline(
        gate_split,
        color="black",
        alpha=0.2
    )

    ax.set_xlabel(r"$\hbar\tau/\Delta$")

    ax.set_ylabel("Relative phase")

    ax.set_title(
        f"{parity.upper()} parity — Relative phase"
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Bloch sphere
    # --------------------------------------------------------

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111, projection='3d')

    setup_bloch(
        ax,
        title=f"{parity.upper()} parity trajectory",
        north=north,
        south=south
    )

    # --------------------------------------------------------
    # split colors
    # --------------------------------------------------------

    split_index = np.argmin(
        np.abs(times - gate_split)
    )

    ax.plot(
        X[:split_index],
        Y[:split_index],
        Z[:split_index],
        lw=4,
        label="first gate"
    )

    ax.plot(
        X[split_index:],
        Y[split_index:],
        Z[split_index:],
        lw=4,
        label="second gate"
    )

    ax.scatter(
        X[0],
        Y[0],
        Z[0],
        s=100
    )

    ax.scatter(
        X[-1],
        Y[-1],
        Z[-1],
        s=150,
        marker="*"
    )

    ax.legend(
        frameon=False
    )

    plt.tight_layout()
    plt.show()

# ============================================================
# RUN SEQUENCES
# ============================================================

plot_sequence(
    times_even,
    psis_even,
    parity="even"
)

plot_sequence(
    times_odd,
    psis_odd,
    parity="odd"
)
