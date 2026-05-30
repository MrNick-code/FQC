import scipy
import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
from sympy.core.singleton import S
import numpy as np

from pymablock.block_diagonalization import block_diagonalize
from sympy.physics.quantum.operatorordering import normal_ordered_form as NO

from second_quantization_selfmade import fermionic_basis_states, matrix_elements


def spinless_model(spins_aligned=True):
    # define symbols
    B, U, mu_L, mu_R, mu_M, eps_M, Delta, u, v = sympy.symbols(
        r"B, U, \mu_L, \mu_R, \mu_M, \epsilon_M, Delta, u, v",
        real=True,
        commutative=True,
        positive=True
    )
    t, alpha_L, alpha_R = sympy.symbols( 
        r"t, alpha_L, alpha_R",
        commutative=True,
        real=True,
        positive=True
    )
    t_L = t*sympy.cos(alpha_L)
    t_R = t*sympy.cos(alpha_R)
    t_so_L = sympy.sin(alpha_L)*sympy.I*t # spin-orbit hopping
    t_so_R = sympy.sin(alpha_R)*sympy.I*t
    symbols = (B, U, mu_L, mu_R, mu_M, eps_M, t, alpha_L, alpha_R, Delta, u, v)

    # define the fermionic operators
    c_L, c_R = FermionOp('c_L'), FermionOp('c_R')
    gamma_up, gamma_down = FermionOp('\gamma_u'), FermionOp('\gamma_d')

    c_M_down = u * gamma_down + v * Dagger(gamma_up)
    c_M_up = u * gamma_up - v * Dagger(gamma_down)

    # write out the hamiltonian in second quantized basis
    H_LR = mu_L * Dagger(c_L) * c_L + mu_R * Dagger(c_R) * c_R
    H_SC = (eps_M - B) * Dagger(gamma_down) * gamma_down + (eps_M + B) * Dagger(gamma_up) * gamma_up
    H_0 = H_LR + H_SC


    if spins_aligned:
        H_p_AB = Dagger(c_L) * ( t_so_L * c_M_down + t_L * c_M_up)
        H_p_AB += (t_R * Dagger(c_M_up) +  t_so_R * Dagger(c_M_down))*c_R
    else:
        H_p_AB = Dagger(c_L) * (t_L * c_M_down + t_so_L * c_M_up)
        H_p_AB += (t_R * Dagger(c_M_up) + t_so_R * Dagger(c_M_down))*c_R

    H_u = U*Dagger(c_L)*c_L*Dagger(c_R)*c_R
    H_p = H_p_AB + Dagger(H_p_AB) + H_u
    H = H_0 + H_p
    H_0 = NO(H_0)
    H_p = NO(H_p)
    H = NO(H)

    basis = fermionic_basis_states([c_L, c_R, gamma_up, gamma_down])
    basis.insert(0, S.One) # vacuum state

    # reorder some things to have AA basis only with c_L and c_R
    ordered_basis = [basis[0]]
    ordered_basis.append(basis[5])
    ordered_basis.extend(basis[1:3])
    ordered_basis.extend(basis[3:5])
    ordered_basis.extend(basis[6:])

    return H_0, H_p, symbols, ordered_basis

def uv_subs(matrix, symbols):
    B, U, mu_L, mu_R, mu_M, eps_M, t, alpha_L, alpha_R, Delta, u, v = symbols
    u_expr = sympy.sqrt(1 + mu_M/eps_M)/sympy.sqrt(2) # I set Delta = 1, so energies are in units of Delta
    v_expr = sympy.sqrt(1 - mu_M/eps_M)/sympy.sqrt(2)
    abs_e = sympy.sqrt(1 + mu_M**2)
    params = {u : u_expr, v : v_expr}
    matrix_subbed = matrix.subs(params).subs({eps_M : abs_e})
    return matrix_subbed


def matrix_hamiltonian(H, basis):
    N = len(basis)
    # calculate all the matrix elements in the many-body basis
    flat_matrix = matrix_elements(H, basis)
    # convert the matrix elements to a sympy matrix
    H_matrix = sympy.Matrix(np.array(flat_matrix).reshape(N, N))
    assert H_matrix.is_hermitian
    return H_matrix

def H_effective(H_0_matrix, H_p_matrix, n):                          
    subspace_indices = np.concatenate((
        np.zeros(n, dtype=int),
        np.ones(H_0_matrix.shape[0] - n, dtype=int)
    ))

    H_tilde, U, U_adj = block_diagonalize(
        [H_0_matrix, H_p_matrix],
        subspace_indices=subspace_indices
    )

    return H_tilde, U, U_adj
    
def symmetric_model(spins_aligned=True, fix_t=False, fix_lr=False):
    H_0, H_p, symbols, ordered_basis = spinless_model(spins_aligned=spins_aligned)
    B, U, mu_L, mu_R, mu_M, eps_M, t, alpha_L, alpha_R, Delta, u, v = symbols

    params = {
        U: U,
        alpha_L : alpha_R,
    }

    if fix_t: # fix_t is used to set down the tunnel parameter when convinient
        params[t] = 0
    if fix_lr:
        params[mu_R] = 0
        params[mu_L] = mu_M
    else:
        params[mu_R] = mu_M
        params[mu_L] = mu_M

    if fix_lr and not fix_t:
        params[mu_L] = .45
        params[mu_R] = .45
        params[mu_L] = .45
        params[mu_R] = .45
    else:
        pass

    H_0_matrix = sympy.simplify(matrix_hamiltonian(H_0, ordered_basis).subs(params))
    H_p_matrix = sympy.simplify(matrix_hamiltonian(H_p, ordered_basis).subs(params))

    # define full numeric hamiltonian
    H_full_n = sympy.lambdify((t, alpha_R, mu_M, B, U),
                        uv_subs(H_0_matrix+H_p_matrix, symbols), 
                        modules=["scipy", {'sqrt' : np.lib.scimath.sqrt}])


    # define the perturbative sub-hamiltonian
    H_tilde, _, _ = H_effective(H_0_matrix, H_p_matrix, 4)
    H_eff = np.sum(H_tilde[0, 0, :3].compressed())
    H_eff = sympy.simplify(uv_subs(H_eff, symbols))
    assert H_eff.is_hermitian
    return (H_eff, symbols), H_full_n

def svd_project(full_ham):
    # projects the hamiltonian into the np.eye(16)[4:, 4:] states
    eigvals, evecs = np.linalg.eigh(full_ham)
    diagonals = np.diag(eigvals[:4])

    # Block diagonalization
    reference_vecs = np.eye(full_ham.shape[0])[:4, :4] ## AA subspace
    overlap_subspaces = evecs[:4, :4].T.conj() @ reference_vecs
    u_svd, _, vh_svd = np.linalg.svd(overlap_subspaces)
    s = u_svd @ vh_svd
    return s.T.conj() @ diagonals @ s