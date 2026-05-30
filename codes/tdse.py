import multiprocessing as mp
import numpy as np
from scipy.integrate import complex_ode as ode
import matplotlib.pyplot as plt
from hamiltonian import svd_project, symmetric_model

if __name__ == "__main__":
    mp.set_start_method("fork")

# get_H_full_n uses the cache to set up fix_t value
_H_CACHE = {}

def get_H_full_n(fix_t=False, fix_lr=False):
    key = ("spins_aligned=False", fix_t, fix_lr)

    if key not in _H_CACHE:
        _, H = symmetric_model(spins_aligned=False, fix_t=fix_t, fix_lr=fix_lr)
        _H_CACHE[key] = H

    return _H_CACHE[key]

def TDSE_solve(dt, t_max, H_t, psi0, args=(), method="vode"):
    """
    Solves the time-dependent Schrodinger equation for a given Hamiltonian
    H_t(t, *args) and initial state psi0.

    Parameters
    ----------
    dt : float
        Time step.
    t_max : float
        Maximum time.
    H_t : function
        Time-dependent Hamiltonian.
    psi0 : array
        Initial state.
    args : tuple, optional
        Additional arguments to be passed to H_t.
    method : string, optional
        Integration method.

    Returns
    -------
    t_eval : array
        Time points at which the solution was evaluated.
    psi : array
        Solution of the TDSE.
    """

    def ode_func(t, psi):
        return -1j * H_t(t, *args) @ psi

    sol = ode(ode_func)
    sol.set_initial_value(psi0)
    sol.set_integrator(method)
    psi = []
    t_eval = []
    while sol.successful() and sol.t < t_max:
        t_eval.append(sol.t + dt)
        psi.append(sol.integrate(sol.t + dt))
    psi = np.array(psi)
    t_eval = np.array(t_eval)
    return t_eval, psi


def sigmoid_pulse(time, time_c, rise_time, pulse_dur):
    """
    Sigmoid pulse with rise time and duration.

    Parameters
    ----------
    time : float
        Time.
    time_c : float
        Center of the pulse.
    rise_time : float
        Rise time.
    pulse_dur : float
        Duration of the pulse.

    Returns
    -------
    float
        Value of the pulse at a given time.
    """
    k = 2 / rise_time
    return 1 / (1 + np.exp(-k * (time - time_c + pulse_dur / 2))) - 1 / (
        1 + np.exp(-k * (time - time_c - pulse_dur / 2))
    )


def TD_pair_pulse(
    pulse_correction, rise_time, t_max=0.3, alpha=np.pi/8, B=0, U=0, psi0=None, method="vode"
):
    H_full_n = get_H_full_n(fix_t=False)
    H_max = H_full_n(t_max, alpha, 0, B, U)
    H_svd = svd_project(H_max)
    CAR_strength = np.abs(H_svd[0, 1])

    if psi0 is None:
        psi0 = np.zeros(16)
        psi0[0] = 1

    pulse_dur = np.pi / (CAR_strength * 4) * pulse_correction
    time_max = 80 # 4 * pulse_dur
    time_c = 35 #time_max / 2
    # dt = pulse_dur / 1000 original
    dt = pulse_dur / 200

    def H_t(time, time_c, rise_time, pulse_dur):
        base_pulse = sigmoid_pulse(time, time_c, rise_time, pulse_dur)
        return H_full_n(t_max * base_pulse, alpha, 0, B, U) # t modulation

    time, psi = TDSE_solve(
        dt, time_max, H_t, psi0, args=(time_c, rise_time, pulse_dur), method=method
    )

    return time, psi, (time_c, rise_time, pulse_dur)

def TD_hop_pulse(
    pulse_correction, rise_time, t_max=0.3, mu_M=0.4, alpha=np.pi/8, B=0, U=0, psi0=None, method="vode"
):
    H_full_n = get_H_full_n(fix_t=False, fix_lr=False)
    H_max = H_full_n(t_max, alpha, mu_M, B, U)
    H_svd = svd_project(H_max)
    ECT_strength = np.abs(H_svd[2, 3])

    if psi0 is None:
        psi0 = np.zeros(16)
        psi0[2] = 1

    pulse_dur = np.pi / (ECT_strength * 4) * pulse_correction
    time_max = 80 # 4 * pulse_dur
    time_c = 35 #time_max / 2
    # dt = pulse_dur / 1000 original
    dt = pulse_dur / 200

    def H_t(time, time_c, rise_time, pulse_dur):
        base_pulse = sigmoid_pulse(time, time_c, rise_time, pulse_dur)
        return H_full_n(t_max * base_pulse, alpha, mu_M, B, U) # t modulation

    time, psi = TDSE_solve(
        dt, time_max, H_t, psi0, args=(time_c, rise_time, pulse_dur), method=method
    )

    return time, psi, (time_c, rise_time, pulse_dur)

# MATHEUS --> u4
def TD_coulomb_pulse(
        U, pulse_correction, rise_time, t_max=0.3, mu_M=0.4, alpha=np.pi/1, B=0, psi0=None, method="vode"
        ):

    # Pulse correction is TAL_p^{(2)} while pulse duration is TAL_p^{(1)}
    # t_max is the final time TAL_2. The rise time rumps up to minimize non-adiabatic transitions

    if psi0 is None:
        psi0 = np.zeros(16)
        psi0[1] = 1/np.sqrt(2)
        psi0[0] = 1/np.sqrt(2)
    
    pulse_dur = np.pi * pulse_correction
    time_max = 12.5714 # 4 * pulse_dur
    time_c = 5.5 #time_max / 2
    # dt = pulse_dur / 1000 original
    dt = pulse_dur / 200

    H_full_n = get_H_full_n(fix_t=True)
    def H_t(time, time_c, rise_time, pulse_dur):
        base_pulse = sigmoid_pulse(time, time_c, rise_time, pulse_dur)
        return H_full_n(t_max, alpha, mu_M, B, U * base_pulse) # U modulation
    
    time, psi = TDSE_solve(
        dt, time_max, H_t, psi0, args=(time_c, rise_time, pulse_dur), method=method
    )

    return time, psi, (time_c, rise_time, pulse_dur)

# MATHEUS --> u1
def TD_phase_pulse(
        pulse_correction, rise_time, t_max=0.3, mu_M=0.4, alpha=np.pi/1, B=0, U=0, psi0=None, method="vode"
):
    if psi0 is None:
        psi0 = np.zeros(16)
        psi0[0] = 1/np.sqrt(2)
        psi0[2] = 1/np.sqrt(2)

    pulse_dur = np.pi * pulse_correction
    time_max = 12.5714
    time_c = 5.5
    dt = pulse_dur / 200

    H_full_n = get_H_full_n(fix_t=True)
    def H_t(time, time_c, rise_time, pulse_dur):
        base_pulse = sigmoid_pulse(time, time_c, rise_time, pulse_dur)
        return H_full_n(t_max, alpha, mu_M * base_pulse, B, U) # mu_M modulation
    
    time, psi = TDSE_solve(
        dt, time_max, H_t, psi0, args=(time_c, rise_time, pulse_dur), method=method
    )

    return time, psi, (time_c, rise_time, pulse_dur)

# U1.U2.U4.U3.|00>
def TD_general_pulse(
    psi0, H_builder, pulse_args, pulse_duration, rise_time, time_offset=0, idle_time_b=0, idle_time_aft=0, dt=0.01, method="dopri5"
    ):

    time_c = pulse_duration / 2
    total_time = idle_time_b + pulse_duration + idle_time_aft

    def H_t_(time):

        if time <= pulse_duration:
            pulse = sigmoid_pulse(
                time,
                time_c,
                rise_time,
                pulse_duration
            )
        else:
            pulse = 0

        return H_builder(pulse, *pulse_args)

    time, psi = TDSE_solve(
        dt,
        total_time,
        lambda t: H_t_(t),
        psi0,
        args=(),
        method=method
    )

    return time + time_offset, psi
