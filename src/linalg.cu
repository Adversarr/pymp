#include "linalg.hpp"

#include <iostream>
#include <mathprim/blas/cublas.cuh>
#include <mathprim/core/defines.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/linalg/iterative/precond/approx_inv.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/precond/eigen_support.hpp>
#include <mathprim/linalg/iterative/precond/ic_cusparse.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/sparse/basic_sparse.hpp>
#include <mathprim/sparse/blas/cusparse.hpp>
#include <mathprim/supports/eigen_sparse.hpp>

using namespace mathprim;

////////////////////////////////////////////////
/// CPU->GPU->CPU
////////////////////////////////////////////////
template <typename Flt,
          typename Precond =
              sparse::iterative::none_preconditioner<Flt, device::cuda>>
std::pair<index_t, double>
cg_cuda(const Eigen::SparseMatrix<Flt, Eigen::RowMajor> &A, //
        nb::ndarray<Flt, nb::shape<-1>, nb::device::cpu> b, //
        nb::ndarray<Flt, nb::shape<-1>, nb::device::cpu> x, //
        const Flt rtol,                                     //
        index_t max_iter,                                   //
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
  auto h_x_view = view(x.data(), make_shape(x.size()));
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
      .norm_tol_ = rtol,
  };
  sparse::iterative::iterative_solver_result<Flt> result;
  auto start = std::chrono::high_resolution_clock::now();
  if (verbose > 0) {
    result = solver.apply(b_view, x_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
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

////////////////////////////////////////////////
/// CUDA direct
////////////////////////////////////////////////
template <typename Flt,
          typename Precond =
              sparse::iterative::none_preconditioner<Flt, device::cuda>>
std::pair<index_t, double> cg_cuda_csr_direct(                         //
    nb::ndarray<index_t, nb::ndim<1>, nb::device::cuda> outer_ptrs,    //
    nb::ndarray<index_t, nb::ndim<1>, nb::device::cuda> inner_indices, //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cuda> values,            //
    index_t rows, index_t cols,                                        //
    nb::ndarray<Flt, nb::shape<-1>, nb::device::cuda> b,               //
    nb::ndarray<Flt, nb::shape<-1>, nb::device::cuda> x,               //
    const Flt rtol,                                                    //
    index_t max_iter,                                                  //
    int verbose) {
  using SparseBlas =
      mp::sparse::blas::cusparse<Flt, sparse::sparse_format::csr>;
  using LinearOp = sparse::iterative::sparse_matrix<SparseBlas>;
  using Blas = blas::cublas<Flt>;
  using Solver =
      mp::sparse::iterative::cg<Flt, mp::device::cuda, LinearOp, Blas, Precond>;
  using SpView = sparse::basic_sparse_view<const Flt, device::cuda, mathprim::sparse::sparse_format::csr>;

  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  if (b.size() != rows || x.size() != cols) {
    throw std::invalid_argument(
        "b and x must have the same size as the matrix.");
  }

  if (max_iter == 0) {
    max_iter = rows;
  }

  const Flt * p_values = values.data();
  const index_t* p_outer = outer_ptrs.data();
  const index_t* p_inner = inner_indices.data();
  const index_t nnz = static_cast<index_t>(values.size());
  if (static_cast<index_t>(outer_ptrs.size()) != rows + 1) {
    throw std::invalid_argument("Invalid outer_ptrs size.");
  }
  if (static_cast<index_t>(inner_indices.size()) != nnz) {
    throw std::invalid_argument("Invalid inner_indices size.");
  }

  SpView view_A(p_values, p_outer, p_inner, rows, cols, nnz, sparse::sparse_property::symmetric);

  Solver solver(LinearOp(view_A), Blas{}, Precond{view_A});

  auto b_view = view<device::cuda>(b.data(), make_shape(b.size())).as_const();
  auto x_view = view<device::cuda>(x.data(), make_shape(x.size()));

  sparse::iterative::iterative_solver_parameters<Flt> criteria{
      .max_iterations_ = max_iter,
      .norm_tol_ = rtol,
  };
  sparse::iterative::iterative_solver_result<Flt> result;
  auto start = std::chrono::high_resolution_clock::now();
  if (verbose > 0) {
    result = solver.apply(b_view, x_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
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
using diagonal = sparse::iterative::diagonal_preconditioner<
    Scalar, device::cuda, sparse::sparse_format::csr, blas::cublas<Scalar>>;

template <typename Scalar>
using ainv = sparse::iterative::approx_inverse_preconditioner<
    sparse::blas::cusparse<Scalar, sparse::sparse_format::csr>>;

template <typename Scalar>
using ic =
    sparse::iterative::cusparse_ichol<Scalar, device::cuda,
                                      mathprim::sparse::sparse_format::csr>;

void bind_linalg_cuda(nb::module_ &m) {
  m.def("cg_cuda", &cg_cuda<float>,
        "Preconditioned Conjugate Gradient method (CUDA).", //
        nb::arg("A"),                                       //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0,
        nb::arg("verbose") = 0);

  m.def("pcg_diagonal_cuda", &cg_cuda<float, diagonal<float>>,
        "Preconditioned Conjugate Gradient method on GPU. (Diagonal "
        "Preconditioner)",                                  //
        nb::arg("A"),                                       //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0,
        nb::arg("verbose") = 0);
  m.def("pcg_ainv_cuda", &cg_cuda<float, ainv<float>>,
        "Preconditioned Conjugate Gradient method on GPU. (Approx Inverse "
        "Preconditioner)",                                  //
        nb::arg("A"),                                       //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0,
        nb::arg("verbose") = 0);
  m.def("cg_cuda", &cg_cuda<double>,
        "Preconditioned Conjugate Gradient method (CUDA).", //
        nb::arg("A"),                                       //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0,
        nb::arg("verbose") = 0);
  m.def("pcg_ic", &cg_cuda<float, ic<float>>,
        "Preconditioned Conjugate Gradient method on GPU. (Incomplete Cholesky "
        "Preconditioner)",                                  //
        nb::arg("A"),                                       //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0,
        nb::arg("verbose") = 0);
  m.def("pcg_diagonal_cuda", &cg_cuda<double, diagonal<double>>,
        "Preconditioned Conjugate Gradient method on GPU. (Diagonal "
        "Preconditioner)",                                  //
        nb::arg("A"),                                       //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0,
        nb::arg("verbose") = 0);
  m.def("pcg_ainv_cuda", &cg_cuda<double, ainv<double>>,
        "Preconditioned Conjugate Gradient method on GPU. (Approx Inverse "
        "Preconditioner)",                                  //
        nb::arg("A"),                                       //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0,
        nb::arg("verbose") = 0);
  m.def("pcg_ic", &cg_cuda<double, ic<double>>,
        "Preconditioned Conjugate Gradient method on GPU. (Incomplete Cholesky "
        "Preconditioner)",                                  //
        nb::arg("A"),                                       //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0,
        nb::arg("verbose") = 0);

  m.def("cg_cuda_csr_direct", &cg_cuda_csr_direct<float>,
        "Preconditioned Conjugate Gradient method (CUDA).",                                                      //
        nb::arg("outer_ptrs").noconvert(), nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(),  //
        nb::arg("rows"), nb::arg("cols"),                                                                        //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(),                                                      //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
  m.def("pcg_cuda_csr_direct_diagonal", &cg_cuda_csr_direct<float, diagonal<float>>,
        "Preconditioned Conjugate Gradient method (CUDA) with Diagonal Preconditioner.",                         //
        nb::arg("outer_ptrs").noconvert(), nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(),  //
        nb::arg("rows"), nb::arg("cols"),                                                                        //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(),                                                      //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
  m.def("pcg_cuda_csr_direct_ic", &cg_cuda_csr_direct<float, ic<float>>,
        "Preconditioned Conjugate Gradient method (CUDA) with Incomplete Cholesky Preconditioner.",              //
        nb::arg("outer_ptrs").noconvert(), nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(),  //
        nb::arg("rows"), nb::arg("cols"),                                                                        //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(),                                                      //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
  m.def("pcg_cuda_csr_direct_ainv", &cg_cuda_csr_direct<float, ainv<float>>,
        "Preconditioned Conjugate Gradient method (CUDA) with Approximate Inverse Preconditioner.",              //
        nb::arg("outer_ptrs").noconvert(), nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(),  //
        nb::arg("rows"), nb::arg("cols"),                                                                        //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(),                                                      //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
  

  m.def("cg_cuda_csr_direct", &cg_cuda_csr_direct<double>,
        "Preconditioned Conjugate Gradient method (CUDA).",                                                      //
        nb::arg("outer_ptrs").noconvert(), nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(),  //
        nb::arg("rows"), nb::arg("cols"),                                                                        //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(),                                                      //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
  m.def("pcg_cuda_csr_direct_diagonal", &cg_cuda_csr_direct<double, diagonal<double>>,
        "Preconditioned Conjugate Gradient method (CUDA) with Diagonal Preconditioner.",                         //
        nb::arg("outer_ptrs").noconvert(), nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(),  //
        nb::arg("rows"), nb::arg("cols"),                                                                        //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(),                                                      //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
  m.def("pcg_cuda_csr_direct_ainv", &cg_cuda_csr_direct<double, ainv<double>>,
        "Preconditioned Conjugate Gradient method (CUDA) with Approximate Inverse Preconditioner.",              //
        nb::arg("outer_ptrs").noconvert(), nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(),  //
        nb::arg("rows"), nb::arg("cols"),                                                                        //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(),                                                      //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
  m.def("pcg_cuda_csr_direct_ic", &cg_cuda_csr_direct<double, ic<double>>,
        "Preconditioned Conjugate Gradient method (CUDA) with Incomplete Cholesky Preconditioner.",              //
        nb::arg("outer_ptrs").noconvert(), nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(),  //
        nb::arg("rows"), nb::arg("cols"),                                                                        //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(),                                                      //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
}
