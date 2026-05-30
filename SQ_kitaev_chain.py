### Standard Kitaev chain in addition to nearest-neighbor Coulomb interactions

# ---------------- imports -----------------------
# Symbolic tools
import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
from IPython.display import display

# Numerical tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Second-quantization package
from second_quantization import hilbert_space
# ------------------------------------------------

# Fermionic annihilation operators c_i for each site in the chain
length = 6
names = [f"c_{i}" for i in range(length)]
fermions = [FermionOp(name) for name in names]

# Symbolic construction of each hamiltonian therm
t, Delta, U = sympy.symbols("t, Delta, U", real=True, positive=True)
mus = sympy.symbols(" ".join(f"mu_{i}" for i in range(length)))

H = t * sum(Dagger(a) * b for a, b in zip(fermions, fermions[1:]))                  # Hopping term: -t Σ (c†ᵢ cᵢ₊₁ + h.c.)
H += Delta * sum(Dagger(a) * Dagger(b) for a, b in zip(fermions, fermions[1:]))     # Pairing term: Δ Σ (c†ᵢ c†ᵢ₊₁ + h.c.)
H += Dagger(H)                                                                      # Add Hermitian conjugate for hopping and pairing
for i, mu in enumerate(mus):
    H += mu * Dagger(fermions[i]) * fermions[i]                                     # Chemical potential: Σ μᵢ c†ᵢ cᵢ
H += U * sum(Dagger(a) * a * Dagger(b) * b for a, b in zip(fermions, fermions[1:])) # Interaction term: U Σ nᵢ nᵢ₊₁ where nᵢ = c†ᵢ cᵢ

display(sympy.Eq(sympy.Symbol("H"), H.factor()))

# H(t, Delta, U, mu_0, ..., mu_7) in matrix form
terms = hilbert_space.to_matrix(H, operators=fermions, sparse=True) # operators to matrix using second_quantization
f_hamiltonian = hilbert_space.make_dict_callable(terms) # callable function H of t, Delta, U, mu_0, ... and mu_7.

# This approach separates the (expensive) symbolic processing from the (cheap) numerical evaluation, making parameter sweeps extremely efficient.!

# Exploiting Symmetry: fermion parity modulo 2 (the total fermions in system are conserved in either odd or even number of total fermions)
parity_operator = hilbert_space.parity_operator(fermions, sparse=True)
#print(parity_operator)

# Parity-resolved diagonalization to reduce computational costs and compute ground states (need to understand better why....)
def fermionic_eigsh(hamiltonian, parity, k=6):
    """
    Diagonalize a Hamiltonian using fermionic superselection rules.
    # The Hamiltonian is assumed to be block-diagonal in the parity subspaces.                                 # parity subspaces?
    The function returns the eigenvalues and eigenvectors of the Hamiltonian, sorted by increasing energy.     # so, E and PSI? (H.PSI=E.PSI)

    Parameters
    ----------
    hamiltonian : scipy.sparse.csr_matrix
        The Hamiltonian matrix to be diagonalized.
    parity : scipy.sparse.csr_matrix
        The parity operator matrix, which is assumed to be diagonal
        in the same basis as the Hamiltonian.
    k : int
        The number of eigenvalues and eigenvectors to compute.
        Default is 6.
    Returns
    -------
    evals : list of numpy.ndarray
        The eigenvalues of the Hamiltonian separated in parity subspaces.
    evecs : list of numpy.ndarray
        The eigenvectors of the Hamiltonian separated in parity subspaces.
    """
    # Separate the Hamiltonian into even and odd parity blocks
    par_mat = parity.diagonal()
    par_mat = 2*(par_mat-1/2)

    odd_inds, even_inds = (
        np.where(par_mat < 0)[0],
        np.where(par_mat > 0)[0],
    )

    h_odd = hamiltonian[odd_inds, :][:, odd_inds]
    h_even = hamiltonian[even_inds, :][:, even_inds]

    off_diagonal = hamiltonian[even_inds, :][:, odd_inds]
    assert off_diagonal.count_nonzero() == 0

    off_diagonal = hamiltonian[odd_inds, :][:, even_inds]
    assert off_diagonal.count_nonzero() == 0

    # Diagonalize each block separately
    evals_even, evecs_even = eigsh(h_even, k=k, which="SA", return_eigenvectors=True, v0=None)
    evals_odd, evecs_odd = eigsh(h_odd, k=k, which="SA", return_eigenvectors=True, v0=None)

    # Combine the eigenvectors into the full space
    new_evecs_even = np.zeros((par_mat.shape[0], evecs_even.shape[1]), dtype=complex)
    new_evecs_odd = np.zeros((par_mat.shape[0], evecs_odd.shape[1]), dtype=complex)

    new_evecs_even[even_inds, :] = evecs_even
    new_evecs_odd[odd_inds, :] = evecs_odd

    # Organise results
    evals = np.array([evals_even, evals_odd])
    evecs = np.array([new_evecs_even, new_evecs_odd])

    return evals, evecs

# About topological protection and phase transitions

# How the energy gap between ground state and first excited state flow based on U and mu?

# parameters
t_val, mu_val, U_values, energies = 1.1, np.linspace(0, 6, 55), np.linspace(-4, 4, 56), []

for U_val in U_values:
    Delta_val = t_val + U_val / 2                                   # Example relation between Delta and U
    for _mu in mu_val:
        param_dict = {'t': t_val, 'Delta': Delta_val, 'U': U_val}   # Build parameter dictionary

        # Set site-dependent chemical potentials (accounting for interaction shift)
        for i, mu in enumerate(mus):
            if i > 0 and i < length - 1:
                param_dict[str(mu)] = _mu - U_val                   # Bulk sites: full interaction correction
            else:
                param_dict[str(mu)] = _mu - U_val / 2               # Edge sites: half interaction correction

        # Generate Hamiltonian and diagonalize
        H_matrix = f_hamiltonian(**param_dict)
        evals, _ = fermionic_eigsh(H_matrix, parity_operator, k=8)
        energies.append(evals)

energies = np.array(energies).reshape(len(U_values), len(mu_val), 2, -1) # reshape
gs_energy = energies[:, :, 1, 0] - energies[:, :, 0, 0] # gap!

# --- plot ---
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.pcolormesh(
    U_values,
    mu_val,
    np.abs(gs_energy.T),
    shading="auto",
    cmap="viridis",
    vmin=0,
    vmax=2
)

ax.set_xlabel(r"$U$")
ax.set_ylabel(r"$\mu$")
ax.set_title(r"$E_{\text{gap}}$")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Energy Gap")
plt.show()
# -----------
