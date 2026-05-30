###  supercurrent between two superconductors weakly coupled through a quantum dot

from IPython.display import display
import numpy as np
import sympy
from sympy import exp, I, symbols, Symbol, Eq
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger

from pymablock import block_diagonalize
from itertools import combinations

import matplotlib.pyplot as plt

# + h.c. --> plus the hermitian conjugate of all therms in the equation!

# --------- Defining the hamiltonian H = Hsc + Hqd + Htc ------------
# Hamiltonian parameters
U, N = symbols(r"U N", positive=True)
xis = xi_L, xi_R = symbols(r"\xi_L \xi_R", positive=True)
Gammas = Gamma_L, Gamma_R = symbols(r"\Gamma_L \Gamma_R", positive=True)
ts = t_L_complex, t_R = Symbol("t_Lc", real=False), Symbol("t_R", real=True)
# Here we use t_L_complex to avoid complications with simplification routines in sympy, so that is not expanded into a sine and cosine

# Dot operators
c_up, c_down = FermionOp(r'c_{d, \uparrow}'), FermionOp(r'c_{d, \downarrow}')

# Superconductor operators
d_ups = FermionOp(r'd_{L, \uparrow}'), FermionOp(r'd_{R, \uparrow}')
d_downs = FermionOp(r'd_{L, \downarrow}'), FermionOp(r'd_{R, \downarrow}')

# Only used for printing. Because sympy forces lexicographic ordering,
# we prepend {} for h.c. to appear last. (.expand())
hc = sympy.Symbol(r"{}\textrm{h.c.}")

def n(op):
    """Shorthand for Dagger(op) * op"""
    return Dagger(op) * op

def display_eq(title, expr):
    """Print a sympy expression as an equality."""
    display(Eq(sympy.Symbol(title), expr))

# Quantum Dot Hamiltonian
H_dot = U * (n(c_up) + n(c_down) - N)**2 / 2

# Tunneling Hamiltonian
H_T = sum(t * (Dagger(c_up) * d_up + Dagger(c_down) * d_down)
    for t, d_up, d_down in zip(ts, d_ups, d_downs)
)  # + h.c. added later

# Hsc isn´t diagonal, but Hdot is. So, applying the Bogoliubov Transformation makes the unperturbed H0 = Hsc + Hdot diagonal!
# Superconductors energies
Es = E_L, E_R = symbols(r"E_L E_R", positive=True)

# Bogoliubov quasiparticle operators
f_ups = FermionOp('f_{L, \\uparrow}'), FermionOp('f_{R, \\uparrow}')
f_downs = FermionOp('f_{L, \\downarrow}'), FermionOp('f_{R, \\downarrow}')

# Superconductors Hamiltonian
H_sc = sum(
    xi - E + E * n(f_up) + E * n(f_down)
    for xi, E, f_up, f_down in zip(xis, Es, f_ups, f_downs)
)

# Now since Ht depends on t, we transform that too
# Bogoliubov coefficients
us = u_L, u_R = symbols(r"u_L u_R", real=True)
vs = v_L, v_R = symbols(r"v_L v_R", real=True)

# Bogoliubov transformation from d operators to f operators
d_subs = {}
for u, v, d_down, d_up, f_down, f_up in zip(us, vs, d_downs, d_ups, f_downs, f_ups):
    d_subs[d_up] = u * f_up - v * Dagger(f_down)
    d_subs[d_down] = u * f_down + v * Dagger(f_up)
    d_subs[Dagger(d_up)] = Dagger(d_subs[d_up])
    d_subs[Dagger(d_down)] = Dagger(d_subs[d_down])

# Substitute d operators with f operators
H_T = H_T.subs(d_subs).expand()  # + h.c., expand to open up parentheses

# --- Total Hamiltonian ---
H = H_sc + H_dot + H_T + Dagger(H_T)

# Here is a similar process of workflow tutorial. Convert to matrix and only do the computations for a few evalues of interest
# copy/paste to_matrix function. Overcomplex to see in detail now........
def to_matrix(H):
    """Compute a matrix representation of a sympy expression with fermion operators."""
    # Add an identity operator to all symbols so that we always work with operators
    H = H.subs({
        s: sympy.physics.quantum.IdentityOperator() * s for s in H.free_symbols
        if not isinstance(s, sympy.physics.quantum.Operator)
    })
    # Choose an order of fermionic operators
    fermions = [
        s for s in H.free_symbols
        if (
            isinstance(s, sympy.physics.quantum.fermion.FermionOp)
            and s.is_annihilation
        )
    ]
    fermions.sort(key=lambda f: f.name.name) # Sort by label to ensure consistent order
    # Compute matrix representations
    s_minus = sympy.Matrix([[0, 1], [0, 0]])
    s_z = sympy.Matrix([[1, 0], [0, -1]])
    s_0 = sympy.eye(2)
    matrix_subs = {
        op: sympy.kronecker_product(*(i * [s_z] + [s_minus] + (len(fermions) - i - 1) * [s_0]))
        for i, op in enumerate(fermions)
    }
    matrix_subs.update({Dagger(op): Dagger(mat) for op, mat in matrix_subs.items()})
    matrix_subs[sympy.physics.quantum.IdentityOperator()] = sympy.eye(2**len(fermions))

    # Generate basis
    basis = [(sympy.S.One,)]
    for n in range(len(fermions)):
        basis.extend(list(combinations(fermions, n + 1)))
    reversed_basis = list(reversed(basis))
    reversed_basis[-1] = (sympy.S.Zero,)

    basis_matrices = []
    for b, nb in zip(basis, reversed_basis):
        expr = [Dagger(op) * op for op in b]
        expr.extend([sympy.physics.quantum.IdentityOperator()-Dagger(op) * op for op in nb])
        basis_matrices.append(sympy.Mul(*expr).expand())
    basis_matrices = [b.subs(matrix_subs, simultaneous=True).expand() for b in basis_matrices]
    basis_order = [np.nonzero(np.array(b.diagonal(), dtype=int)[0])[0][0] for b in basis_matrices]
    basis = [sympy.Mul(*basis[i]) for i in np.argsort(basis_order)]
    return H.subs(matrix_subs, simultaneous=True).expand(), basis
# --------------

# Compute Hamiltonian in matrix form
H, basis = to_matrix(H)

# To make the denominater of the equation simpler, we take a few steps:
t_L, phi, dphi = symbols(r"t_L \phi \delta\phi", positive=True)
H = H.subs({t_L_complex: t_L * exp(I * (phi + dphi))})

E_0, E_1, E_2 = symbols(r"E_0 E_1 E_2", positive=True)
E_0_value, E_1_value, E_2_value = [(U * (N - i)**2 / 2).expand() for i in range(3)]
# Using the dot energies En right away and introducing a perturbative parameter delta_phi to  the phase between the 2 superconductors 
# The diagonal elements of H correspond to the dot energies En. To compute En we need to take dH_effective/dphi (Heff_phi)
# Again, we take the even and odd parities as subspaces of H (block-diagonal H)

# FInal hamiltonian for further analysis (not effective hamiltonian yet)
H = H.subs({E_2_value: E_2}).subs({E_1_value: E_1}).subs({E_0_value: E_0}) 




############### Daqui pra frente virou um caos, ja não entendi mais nada............... !!!!!!!!!!!!!!

values = {
    U: 10,
    Gamma_L: 0.01,
    Gamma_R: 0.01,
    t_L: 0.4,
    t_R: 0.1,
    xi_L: 0.2,
    xi_R: 0.1,
    E_0: E_0_value,
    E_1: E_1_value,
    E_2: E_2_value,
    phi: np.pi/4,
    dphi: 0,
}
num = 180

### Computing supercurrent: I = (e/h') * dE/dphi

# ???
"""we need to find the perturbed ground state energy E(phi). To do so, we finally use Pymablock to compute the perturbative
 corrections to the ground state Hamiltonian. Because we are interested in different ground states, we define a separate subspace for n = 0, 1, 2. 
 Additionally, to take advantage of the block-diagonal structure of the Hamiltonian, we define separate subspaces for even and odd parity sectors.
 This way, Pymablock avoids computing the matrix elements between states with different parities because they are zero.
 We handle the five subspaces separately by labeling each basis state in the input to the block-diagonalization
 routine with the corresponding subspace number."""

ground_states = [sympy.S.One, c_up, c_down, c_up * c_down]  # vacuum state
subspaces = {
    sympy.S.One: 0,  # n=0
    c_up: 1,  # n=1
    c_down: 1,  # n=1
    c_down * c_up: 2,  # n=2
}
subspace_indices = [
    subspaces.get(element, 3 if len(element.free_symbols) % 2 else 4) for element in basis
]  # 3 for odd, 4 for even
H_tilde = block_diagonalize(H, subspace_indices=subspace_indices, symbols=[t_L, t_R, dphi])[0] 

current = H_tilde[0, 0, 2, 2, 1][0, 0].subs({dphi: 1})

# Simplifying this crazy equation for current
# Define Bogoliubov substitutions
subs = (
    {u * v: Gamma / (2 * E) for u, v, Gamma, E in zip(us, vs, Gammas, Es)}
    | {u**2: (1 + xi / E) / 2 for u, xi, E in zip(us, xis, Es)}
    | {v**2: (1 - xi / E) / 2 for v, xi, E in zip(vs, xis, Es)}
)

def simplify_current(expr):
    """Simplification routine tailored to the perturbative calculation of current."""
    return sympy.re(expr.factor()).simplify().subs(subs)

current = simplify_current(current)

display_eq('I(n=0)', current)

# same procedurer for others ground states:
currents = [current]
currents.extend(
    simplify_current(H_tilde[i, i, 2, 2, 1][0, 0]).subs({dphi: 1}).doit()
    for i in range(1, 3)
)
for i, current in enumerate(currents):
    display_eq(f"I(n={i})", current)

# --- plot: critical current as function of number of electrons ---
N_values = np.linspace(-0.5, 2.5, num)
current_values = [np.array([current.subs({**values, N: N_value}) for N_value in N_values], dtype=float) for current in currents]

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(N_values, np.abs(current_values[0]), '-', label=r'$n=0$')
ax.plot(N_values, np.abs(current_values[1]), '-', label=r'$n=1$')
ax.plot(N_values, np.abs(current_values[2]), '-', label=r'$n=2$')
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$I_c$')
ax.set_title(r'Critical current')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([r'$0$', r'$1$', r'$2$'])
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.show()
# ---
