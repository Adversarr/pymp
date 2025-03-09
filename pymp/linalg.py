from typing import Tuple, List, Union
import libpymp
import numpy as np
from scipy.sparse import csr_matrix


def cg(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.cg(A, b, x, rtol, max_iter, verbose)

def pcg_diagonal(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with diagonal preconditioner.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.pcg_diagonal(A, b, x, rtol, max_iter, verbose)

def pcg_ainv(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Approximated Inverse preconditioner.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.pcg_ainv(A, b, x, rtol, max_iter, verbose)

def cg_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method on GPU.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.cg_cuda(A, b, x, rtol, max_iter, verbose)

def pcg_diagonal_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with diagonal preconditioner on GPU.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.pcg_diagonal_cuda(A, b, x, rtol, max_iter, verbose)

def pcg_ainv_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Approximated Inverse preconditioner on GPU.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.pcg_ainv_cuda(A, b, x, rtol, max_iter, verbose)

def ainv_content(A: csr_matrix) -> csr_matrix:
    """
    Compute the content of the Approximated Inverse preconditioner.

    Returns
    -------
    csr_matrix
        The content of the Approximated Inverse preconditioner.
    """
    return libpymp.linalg.ainv_content(A)


def grid_laplacian_nd_dbc(
    grids: Union[List[int], np.ndarray],
    dtype=np.float32
) -> csr_matrix:
    """
    Construct the Laplacian operator on n-dimensional grid with Dirichlet boundary conditions.

    Returns
    -------
    csr_matrix
        The Laplacian operator.
    """

    if isinstance(grids, np.ndarray):
        grids = grids.tolist()

    if dtype == np.float32:
        return libpymp.linalg.grid_laplacian_nd_dbc_float32(grids)
    elif dtype == np.float64:
        return libpymp.linalg.grid_laplacian_nd_dbc_float64(grids)
    else:
        raise ValueError("dtype must be np.float32 or np.float64")
