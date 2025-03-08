from pymp.linalg import *

import numpy as np
from scipy.sparse import csr_matrix
from timeit import Timer

x = np.array([1,2,3], dtype=np.float32)
b = np.array([3,2,1], dtype=np.float32)
A = csr_matrix(
  [[3, 1, 0],
   [1, 3, 1],
   [0, 1, 3]], dtype=np.float32)


print(cg(A, b, x, 1e-4, 20, 1))
print(x)
print(A @ x - b)

x = np.array([1,2,3], dtype=np.float32)
print(cg_cuda(A, b, x, 1e-4, 20, 1))
print(x)
print(A @ x - b)

for method in [cg, cg_cuda, pcg_diagonal, pcg_ainv, pcg_diagonal_cuda, pcg_ainv_cuda]:
    print(method.__name__)
    x = np.array([1,2,3], dtype=np.float32)
    print(method(A, b, x, 1e-4, 20, 1))
    print(x)
    print(A @ x - b)


def laplace_2d(n) -> csr_matrix:
    """
    Construct the 2D Laplacian matrix of size n x n.
    """
    row_entry = []
    col_entry = []
    data_entry = []
    for i in range(n):
        for j in range(n):
            row_entry.append(i*n + j)
            col_entry.append(i*n + j)
            data_entry.append(4)
            if i > 0:
                row_entry.append(i*n + j)
                col_entry.append((i-1)*n + j)
                data_entry.append(-1)
            if i < n-1:
                row_entry.append(i*n + j)
                col_entry.append((i+1)*n + j)
                data_entry.append(-1)
            if j > 0:
                row_entry.append(i*n + j)
                col_entry.append(i*n + j-1)
                data_entry.append(-1)
            if j < n-1:
                row_entry.append(i*n + j)
                col_entry.append(i*n + j+1)
                data_entry.append(-1)
    res = csr_matrix((data_entry, (row_entry, col_entry)), shape=(n*n, n*n), dtype=np.float32)
    res.sort_indices()
    return res

n = 1024
cnt = 3
A = laplace_2d(n)


def eval_once(solver):
    x = np.ones(n*n, dtype=np.float32)
    b = A @ x
    x = np.zeros(n*n, dtype=np.float32)
    it, er = solver(A, b, x, 1e-6, n * n * 4, 0)
    print(it, er)

for method in [cg, cg_cuda, pcg_diagonal, pcg_ainv, pcg_diagonal_cuda, pcg_ainv_cuda]:
    print(method.__name__)
    t = Timer(lambda: eval_once(method))
    print(t.timeit(number=cnt) / cnt)