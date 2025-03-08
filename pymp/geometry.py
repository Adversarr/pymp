import libpymp
import numpy as np
from scipy.sparse import csr_matrix

def laplacian(vert: np.ndarray, edge: np.ndarray) -> csr_matrix:
    """
    Construct the Laplacian matrix of a mesh.
    """
    return libpymp.geometry.laplacian(vert, edge)

def lumped_mass(vert: np.ndarray, edge: np.ndarray) -> csr_matrix:
    """
    Construct the lumped mass matrix of a mesh.
    """
    return libpymp.geometry.lumped_mass(vert, edge)