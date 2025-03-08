#include "linalg.hpp"

#include <iostream>
#include <mathprim/blas/cpu_blas.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/precond/approx_inv.hpp>
#include <mathprim/linalg/iterative/precond/eigen_support.hpp>

#include <mathprim/sparse/blas/eigen.hpp>

using namespace mathprim;

template <typename Flt, typename Precond = sparse::iterative::none_preconditioner<Flt, device::cpu>>
std::pair<index_t, double> cg_host(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,             //
                                   py::array_t<Flt, py::array::c_style | py::array::forcecast> b,  //
                                   py::array_t<Flt, py::array::c_style | py::array::forcecast> x,  //
                                   const Flt rtol,                                                 //
                                   index_t max_iter,                                               //
                                   int verbose) {
  using SparseBlas = mp::sparse::blas::eigen<Flt, sparse::sparse_format::csr>;
  using LinearOp = sparse::iterative::sparse_matrix<SparseBlas>;
  using Blas = mp::blas::cpu_blas<Flt>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cpu, LinearOp, Blas, Precond>;

  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  if (b.size() != A.rows() || x.size() != A.cols()) {
    throw std::invalid_argument("b and x must have the same size as the matrix.");
  }

  auto b_view = view(b.data(), make_shape(b.size()));
  auto x_view = view(x.mutable_data(), make_shape(x.size()));
  if (max_iter == 0) {
    max_iter = A.rows();
  }
  auto view_A = eigen_support::view(A);
  Solver solver(LinearOp(view_A), Blas{}, Precond{view_A});
  sparse::iterative::iterative_solver_parameters<Flt> criteria {
    .max_iterations_ = max_iter,
    .norm_tol_ = rtol,
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
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  double seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(duration).count();
  return std::make_pair(result.iterations_, seconds);
}

template <typename Scalar>
using diagonal = sparse::iterative::diagonal_preconditioner<Scalar, device::cpu, sparse::sparse_format::csr, blas::cpu_blas<Scalar>>;

template <typename Scalar>
using ainv = sparse::iterative::approx_inverse_preconditioner<sparse::blas::eigen<Scalar, sparse::sparse_format::csr>>;

template <typename Scalar>
using ic = sparse::iterative::eigen_ichol<Scalar>;

void bind_linalg(py::module_& m) {
  m.def("cg", &cg_host<float>, "Preconditioned Conjugate Gradient method on CPU.",  //
        py::arg("A"),                                                               //
        py::arg("b").noconvert(), py::arg("x").noconvert(),                         //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0, py::arg("verbose") = 0);

  m.def("pcg_diagonal", &cg_host<float, diagonal<float>>,
        "Preconditioned Conjugate Gradient method on CPU. (Diagonal Preconditioner)",  //
        py::arg("A"),                                                                  //
        py::arg("b").noconvert(), py::arg("x").noconvert(),                            //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0, py::arg("verbose") = 0);
  m.def("pcg_ainv", &cg_host<float, ainv<float>>,
        "Preconditioned Conjugate Gradient method on CPU. (Approx Inverse Preconditioner)",  //
        py::arg("A"),                                                                        //
        py::arg("b").noconvert(), py::arg("x").noconvert(),                                  //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0, py::arg("verbose") = 0);
  m.def("pcg_ic", &cg_host<float, ic<float>>,
        "Preconditioned Conjugate Gradient method on CPU. (Incomplete Cholesky Preconditioner)",  //
        py::arg("A"),                                                                             //
        py::arg("b").noconvert(), py::arg("x").noconvert(),                                       //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0, py::arg("verbose") = 0);


  m.def("cg", &cg_host<double>, "Preconditioned Conjugate Gradient method on CPU.",  //
        py::arg("A"),                                                                //
        py::arg("b").noconvert(), py::arg("x").noconvert(),                          //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0, py::arg("verbose") = 0);

  m.def("pcg_diagonal", &cg_host<double, diagonal<double>>,
        "Preconditioned Conjugate Gradient method on CPU. (Diagonal Preconditioner)",  //
        py::arg("A"),                                                                  //
        py::arg("b").noconvert(), py::arg("x").noconvert(),                            //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0, py::arg("verbose") = 0);
  m.def("pcg_ainv", &cg_host<double, ainv<double>>,
        "Preconditioned Conjugate Gradient method on CPU. (Approx Inverse Preconditioner)",  //
        py::arg("A"),                                                                        //
        py::arg("b").noconvert(), py::arg("x").noconvert(),                                  //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0, py::arg("verbose") = 0);
  m.def("pcg_ic", &cg_host<double, ic<double>>,
        "Preconditioned Conjugate Gradient method on CPU. (Incomplete Cholesky Preconditioner)",  //
        py::arg("A"),                                                                             //
        py::arg("b").noconvert(), py::arg("x").noconvert(),                                       //
        py::arg("rtol") = 1e-4f, py::arg("max_iter") = 0, py::arg("verbose") = 0);
}
#ifndef MATHPRIM_ENABLE_CUDA
void bind_linalg_cuda(py::module_& m) {
  // Do nothing
}
#endif