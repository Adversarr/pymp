#include "linalg.hpp"

#include <iostream>
#include <mathprim/blas/cublas.cuh>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/linalg/iterative/precond/approx_inv.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/precond/eigen_support.hpp>
#include <mathprim/linalg/iterative/precond/ic_cusparse.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/sparse/blas/cusparse.hpp>
#include <mathprim/supports/eigen_sparse.hpp>

using namespace mathprim;

template <typename Flt,
          typename Precond =
              sparse::iterative::none_preconditioner<Flt, device::cuda>>
std::pair<index_t, double>
cg_cuda(const Eigen::SparseMatrix<Flt, Eigen::RowMajor> &A,            //
        py::array_t<Flt, py::array::c_style | py::array::forcecast> b, //
        py::array_t<Flt, py::array::c_style | py::array::forcecast> x, //
        const Flt atol,                                                //
        index_t max_iter,                                              //
        int verbose) {
  using SparseBlas =
      mp::sparse::blas::cusparse<Flt, sparse::sparse_format::csr>;
  using LinearOp = sparse::iterative::sparse_matrix<SparseBlas>;
  using Blas = blas::cublas<Flt>;
  using Solver =
      mp::sparse::iterative::cg<Flt, mp::device::cuda, LinearOp, Blas, Precond>;

  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  if (b.size() != A.rows() || x.size() != A.cols()) {
    throw std::invalid_argument(
        "b and x must have the same size as the matrix.");
  }

  auto h_b_view = view(b.data(), make_shape(b.size()));
  auto h_x_view = view(x.mutable_data(), make_shape(x.size()));
  if (max_iter == 0) {
    max_iter = A.rows();
  }
  auto d_b = make_cuda_buffer<Flt>(b.size());
  auto d_x = make_cuda_buffer<Flt>(x.size());
  auto b_view = d_b.view();
  auto x_view = d_x.view();
  auto view_A = eigen_support::view(A);

  auto d_A = sparse::basic_sparse_matrix<Flt, device::cuda,
                                         mathprim::sparse::sparse_format::csr>(
      view_A.rows(), view_A.cols(), view_A.nnz());

  copy(b_view, h_b_view);
  copy(x_view, h_x_view);

  copy(d_A.outer_ptrs().view(), view_A.outer_ptrs());
  copy(d_A.inner_indices().view(), view_A.inner_indices());
  copy(d_A.values().view(), view_A.values());

  auto d_A_view = d_A.const_view();
  Solver solver(LinearOp(d_A_view), Blas{}, Precond{d_A_view});
  sparse::iterative::iterative_solver_parameters<Flt> criteria{
      .max_iterations_ = max_iter,
      .norm_tol_ = atol,
  };
  sparse::iterative::iterative_solver_result<Flt> result;
  auto start = std::chrono::high_resolution_clock::now();
  if (verbose > 0) {
    result = solver.apply(b_view, x_view, criteria, [](index_t iter, Flt norm) {
      std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
    });
  } else {
    result = solver.apply(b_view, x_view, criteria);
  }

  copy(h_x_view, x_view);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  double seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(duration).count();
  return std::make_pair(result.iterations_, seconds);
}

template <typename Scalar>
using diagonal = sparse::iterative::diagonal_preconditioner<
    Scalar, device::cuda, sparse::sparse_format::csr, blas::cublas<Scalar>>;

template <typename Scalar>
using ainv = sparse::iterative::approx_inverse_preconditioner<
    sparse::blas::cusparse<Scalar, sparse::sparse_format::csr>>;

template <typename Scalar>
using ic =
    sparse::iterative::cusparse_ichol<Scalar, device::cuda,
                                      mathprim::sparse::sparse_format::csr>;

void bind_linalg_cuda(py::module_ &m) {
  m.def("cg_cuda", &cg_cuda<float>,
        "Preconditioned Conjugate Gradient method (CUDA).", //
        py::arg("A"),                                       //
        py::arg("b").noconvert(), py::arg("x").noconvert(), //
        py::arg("atol") = 1e-4f, py::arg("max_iter") = 0,
        py::arg("verbose") = 0);

  m.def("pcg_diagonal_cuda", &cg_cuda<float, diagonal<float>>,
        "Preconditioned Conjugate Gradient method on GPU. (Diagonal "
        "Preconditioner)",                                  //
        py::arg("A"),                                       //
        py::arg("b").noconvert(), py::arg("x").noconvert(), //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0,
        py::arg("verbose") = 0);
  m.def("pcg_ainv_cuda", &cg_cuda<float, ainv<float>>,
        "Preconditioned Conjugate Gradient method on GPU. (Approx Inverse "
        "Preconditioner)",                                  //
        py::arg("A"),                                       //
        py::arg("b").noconvert(), py::arg("x").noconvert(), //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0,
        py::arg("verbose") = 0);
  m.def("cg_cuda", &cg_cuda<double>,
        "Preconditioned Conjugate Gradient method (CUDA).", //
        py::arg("A"),                                       //
        py::arg("b").noconvert(), py::arg("x").noconvert(), //
        py::arg("atol") = 1e-4f, py::arg("max_iter") = 0,
        py::arg("verbose") = 0);
  m.def("pcg_ic", &cg_cuda<float, ic<float>>,
        "Preconditioned Conjugate Gradient method on GPU. (Incomplete Cholesky "
        "Preconditioner)",                                  //
        py::arg("A"),                                       //
        py::arg("b").noconvert(), py::arg("x").noconvert(), //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0,
        py::arg("verbose") = 0);
  m.def("pcg_diagonal_cuda", &cg_cuda<double, diagonal<double>>,
        "Preconditioned Conjugate Gradient method on GPU. (Diagonal "
        "Preconditioner)",                                  //
        py::arg("A"),                                       //
        py::arg("b").noconvert(), py::arg("x").noconvert(), //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0,
        py::arg("verbose") = 0);
  m.def("pcg_ainv_cuda", &cg_cuda<double, ainv<double>>,
        "Preconditioned Conjugate Gradient method on GPU. (Approx Inverse "
        "Preconditioner)",                                  //
        py::arg("A"),                                       //
        py::arg("b").noconvert(), py::arg("x").noconvert(), //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0,
        py::arg("verbose") = 0);
//   m.def("pcg_ic", &cg_cuda<double, ic<double>>,
//         "Preconditioned Conjugate Gradient method on GPU. (Incomplete Cholesky "
//         "Preconditioner)",                                  //
//         py::arg("A"),                                       //
//         py::arg("b").noconvert(), py::arg("x").noconvert(), //
//         py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0,
//         py::arg("verbose") = 0);
}