import numpy as np
import scipy as sp
from scipy import sparse
from pyhops.dynamics.hops_system import HopsSystem as HSystem
from pyhops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp

__title__ = "test for System Class"
__author__ = "D. I. G. Bennett, Leonel Varvelo"
__version__ = "0.1"
__date__ = "Jan. 15, 2020"

# HOPS SYSTEM PARAMETERS
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 25000.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
}

nsite = 4
e_lambda = 20.0
gamma = 50.0
temp = 140.0
(g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

loperator = np.zeros([4, 4, 4], dtype=np.float64)
gw_sysbath = []
lop_list = []
for i in range(nsite):
    loperator[i, i, i] = 1.0
    gw_sysbath.append([g_0, w_0])
    lop_list.append(sp.sparse.coo_matrix(loperator[i]))
    gw_sysbath.append([-1j * np.imag(g_0), 500.0])
    lop_list.append(loperator[i])

hs = np.zeros([nsite, nsite])
hs[0, 1] = 40
hs[1, 0] = 40
hs[1, 2] = 10
hs[2, 1] = 10
hs[2, 3] = 40
hs[3, 2] = 40

sys_param = {
    "HAMILTONIAN": np.array(hs, dtype=np.complex128),
    "GW_SYSBATH": gw_sysbath,
    "L_HIER": lop_list,
    "L_NOISE1": lop_list,
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": gw_sysbath,
}
HS = HSystem(sys_param)


def test_reduce_sparse_matrix():
    """
    This function test to make sure reduced_sparse_matrix is properly taking in a
    sparse matrix and list which represents the absolute state and converting it to a
    new relative state represented in a sparse matrix
    """
    state_list = [0, 1, 3]
    full_matrix = lop_list[6]
    coo_matrix = sp.sparse.coo_matrix(full_matrix)
    coo_matrix = HS.reduce_sparse_matrix(coo_matrix, state_list)
    coo_matrix = coo_matrix.todense()
    known_matrix = np.zeros((3, 3))
    known_matrix[2, 2] = 1
    assert np.array_equal(coo_matrix, np.array(known_matrix))