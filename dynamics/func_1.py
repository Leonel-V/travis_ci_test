import numpy as np
import scipy as sp
from scipy import sparse


def reduce_sparse_matrix(coo_mat, state_list):
    """
    This function takes in a sparse matrix and list which represents the absolute
    state to a new relative state represented in a sparse matrix

    NOTE: This function assumes that all states associated with the non-zero
    elements of the sparse matrix are present in the state list!

    PARAMETERS
    ----------
    1. coo_mat : np.array
                 a sparse matrix
    2. state_list : list
                   a list of relative index
    RETURNS
    -------
    1. sparse : np.array
                sparse matrix in relative basis
    """
    coo_row = [list(state_list).index(i) for i in coo_mat.row]
    coo_col = [list(state_list).index(i) for i in coo_mat.col]
    coo_data = coo_mat.data
    return sp.sparse.coo_matrix(
        (coo_data, (coo_row, coo_col)), shape=(len(state_list), len(state_list))
    )