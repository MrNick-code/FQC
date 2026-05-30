import pymablock as pb
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
import numpy.ma as ma
from scipy.linalg import block_diag
from scipy.sparse import block_diag as block_diag2

# Defining Hamiltonian (with no pertubation)
H0 = np.diag([-1, -1, 1, 1])

# Adding random pertubation
def random_H(n):
	H = np.random.randn(n, n) + 1j * np.random.randn(n, n)
	H += H.conj().T
	return H

H1 = 0.2 * random_H(4)

# ---
"""
fig, (ax_0, ax_1) = plt.subplots(ncols=2)

ax_0.imshow(H0.real, cmap='seismic', vmin=-2, vmax=2)
ax_0.set_title(r'Unperturbed Hamiltonian $H_0$')
ax_0.set_xticks([])
ax_0.set_yticks([])

ax_1.imshow(H1.real, cmap='seismic', vmin=-2, vmax=2)
ax_1.set_title(r'Perturbation $H_1$')
ax_1.set_xticks([])
ax_1.set_yticks([]);
plt.show()
"""
# ---

# Pertubation theory solution
hamiltonian = [H0, H1]
H_til, U, U_adj = pb.block_diagonalize(hamiltonian, subspace_indices=[0, 0, 1, 1])
# block_diagonalize ~ It takes all the possible types of input and defines a solution of the perturbation theory problem
# 	as infinite series of the transformed Hamiltonian H~ and the corresponding transformation U.
# The subspace_indices argument defines to which subspace each diagonal term of belongs
energy = pb.series.cauchy_dot_product(U_adj, H_til, U)
print(f"H = H0 + H1. 0th order Energy (00): {energy[0, 0, 0]} / 0th order Energy (11): {energy[1, 1, 0]}") 

# ---
"""
# Perturbative results: H_tilde[i, j, order_of_correction]
print(H_til[0, 0, 2])
print("-------------")
print(type(H_til[1, 1, :3]), H_til[1, 1, :3])
# Masked Arrays (instead of arrays) are used to skip zero therms in calculations

transformed_H = ma.sum(H_til[:2, :2, :3], axis=2)
block = block_diag(transformed_H[0, 0], transformed_H[1, 1])

fix, ax_2 = plt.subplots()
ax_2.imshow(block.real, cmap='seismic', vmin=-2, vmax=2)
ax_2.set_title(r'Transformed Hamiltonian $\tilde{H}$')
ax_2.set_xticks([])
ax_2.set_yticks([])
plt.show()
"""
# --- 

# General Hamiltonians -- Ex: H = H00 + lamb1.H10 + (lamb1)²H20 + lamb2.H01
H_00 = random_H(7)  # Unperturbed Hamiltonian
H_10 = 0.1 * random_H(7)  # Linear term in the first perturbative parameter
H_20 = 0.1 * random_H(7)  # Quadratic term in the first perturbative parameter
H_01 = 0.1 * random_H(7)  # Linear term in the second perturbative parameter
hamiltonian2 = {(0, 0): H_00, (1, 0): H_10, (2, 0): H_20, (0, 1): H_01}
# The keys of the hamiltonian dictionary are tuples of integers, where -th integer is the order of the -th perturbative parameter.

# ---
"""
# H00 Isn´t diagonal anymore
plt.figure(figsize=(3, 3))
plt.imshow(H_00.real, cmap='seismic', vmin=-2, vmax=2)
plt.title(r'Unperturbed $H_{00}$')
plt.xticks([])
plt.yticks([])
plt.show()
"""
# ---

# Defining subspaces of H_til using H00 as reference
_, evecs = np.linalg.eigh(H_00)
subspace_ev = [evecs[:, :3], evecs[:, 3:6], evecs[:, 6:]] # 3 subspaces !!!
H_til2, U2, U_adj2 = pb.block_diagonalize(hamiltonian2, subspace_eigenvectors=subspace_ev)

# ---
"""
H_0 = block_diag2(H_til2[[0, 1, 2], [0, 1, 2], 0, 0]).toarray()

fix, ax_2 = plt.subplots()
ax_2.imshow(H_0.real, cmap='seismic', vmin=-2, vmax=2)
ax_2.set_title(r'$H_{00}$ in its eigenbasis')
ax_2.set_xticks([])
ax_2.set_yticks([])
plt.show()

print(f"H~ type: {type(H_til2)}, shape: {H_til2.shape}")

transformed_H = ma.sum(H_til2[:3, :3, :2, :2], axis=(2, 3))
block = block_diag(transformed_H[0, 0], transformed_H[1, 1], transformed_H[2, 2])

# second order perturbative plot
fix, ax_2 = plt.subplots()
ax_2.imshow(block.real, cmap='seismic', vmin=-2, vmax=2)
ax_2.set_title(r'Transformed Hamiltonian $\tilde{H}$')
ax_2.set_xticks([])
ax_2.set_yticks([])
plt.show()
"""
# ---

# aditional operator X as basis of H_til3
H_0_ = random_H(4)
H_1_ = 0.1 * random_H(4)
X = random_H(4)
__, evecs_ = np.linalg.eigh(H_0_)

H_til3, U3, U_adj3 = pb.block_diagonalize([H_0_, H_1_], subspace_eigenvectors=[evecs_[:, :2], evecs_[:, 2:]])

# putting X into same H_0_ basis, perturbative parameters and block structure
X_series = pb.operator_to_BlockSeries({(0,): X}, name="X", hermitian=True, subspace_eigenvectors=[evecs_[:, :2], evecs_[:, 2:]])
X_til3 = pb.series.cauchy_dot_product(U_adj3, X_series, U3)

assert np.allclose(X_til3[0, 0, 0], X_series[0, 0, 0]), "assertion between zeroth order of X failed!"
print(f"Assertion successful: {X_til3[0, 0, 0]} is the same as {X_series[0, 0, 0]}")
print("---")
print(X_til3[0, 0, 2])

### Masking

# hamiltonian
N = 16
h_0 = np.diag(np.arange(N) + 0.2 * np.random.randn(N))
h_1 = 0.1 * np.random.randn(N, N)
h_1 += h_1.T

# Defining mask
smiley_binary = np.array(
    [
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ],
    dtype=bool,
)

mask = np.zeros((N, N), dtype=bool)
mask[1:smiley_binary.shape[0] + 1, -smiley_binary.shape[1] - 1:-1] = smiley_binary
mask = ~(mask | mask.T)
np.fill_diagonal(mask, False)

h_tilde, *_ = pb.block_diagonalize([h_0, h_1], fully_diagonalize=mask)

# plot
Heff = h_tilde[0, 0, 0] + h_tilde[0, 0, 1] + h_tilde[0, 0, 2]

plt.imshow(Heff, cmap='seismic', norm=TwoSlopeNorm(vcenter=0))
plt.title(r'$\tilde{H}$')
plt.xticks([])
plt.yticks([])
plt.show()
