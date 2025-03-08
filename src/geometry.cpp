#include "geometry.hpp"


#include <Eigen/Sparse>
#include <iostream>

#include "mathprim/geometry/laplacian.hpp"
#include "mathprim/geometry/lumped_mass.hpp"
#include "mathprim/supports/eigen_sparse.hpp"

template <typename Flt>
Eigen::SparseMatrix<Flt, Eigen::RowMajor> laplacian(
    py::array_t<Flt, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int, py::array::c_style | py::array::forcecast> faces) {
  if (vertices.ndim() != 2) {
    throw std::runtime_error("vertices must be 2D array");
  }
  if (faces.ndim() != 2) {
    throw std::runtime_error("faces must be 2D array");
  }

  auto nvert = static_cast<mp::index_t>(vertices.shape(0));
  auto nface = static_cast<mp::index_t>(faces.shape(0));
  auto ndim = static_cast<mp::index_t>(vertices.shape(1));
  auto dsimplex = static_cast<mp::index_t>(faces.shape(1));

  using namespace mp::literal;
  if (ndim == 2 && dsimplex == 3) {
    mp::geometry::basic_mesh<Flt, 2, 3, mp::device::cpu> mesh{
      mp::view(vertices.data(), mp::make_shape(nvert, 2_s)),
      mp::view(faces.data(), mp::make_shape(nface, 3_s))};
    auto matrix = mp::geometry::build_laplacian<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else if (ndim == 3 && dsimplex == 3) {
    mp::geometry::basic_mesh<Flt, 3, 3, mp::device::cpu> mesh{
      mp::view(vertices.data(), mp::make_shape(nvert, 3_s)),
      mp::view(faces.data(), mp::make_shape(nface, 3_s))};
    auto matrix = mp::geometry::build_laplacian<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else if (ndim == 3 && dsimplex == 4) {
    mp::geometry::basic_mesh<Flt, 3, 4, mp::device::cpu> mesh{
      mp::view(vertices.data(), mp::make_shape(nvert, 3_s)),
      mp::view(faces.data(), mp::make_shape(nface, 4_s))};
    auto matrix = mp::geometry::build_laplacian<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else {
    std::ostringstream oss;
    oss << "Unsupported dimension or simplex type: ndim=" << ndim << ", dsimplex=" << dsimplex;
    throw std::runtime_error(oss.str());
  }
}


template <typename Flt>
Eigen::SparseMatrix<Flt, Eigen::RowMajor> lumped_mass(
    py::array_t<Flt, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int, py::array::c_style | py::array::forcecast> faces) {
  if (vertices.ndim() != 2) {
    throw std::runtime_error("vertices must be 2D array");
  }
  if (faces.ndim() != 2) {
    throw std::runtime_error("faces must be 2D array");
  }

  auto nvert = static_cast<mp::index_t>(vertices.shape(0));
  auto nface = static_cast<mp::index_t>(faces.shape(0));
  auto ndim = static_cast<mp::index_t>(vertices.shape(1));
  auto dsimplex = static_cast<mp::index_t>(faces.shape(1));

  using namespace mp::literal;
  if (ndim == 2 && dsimplex == 3) {
    mp::geometry::basic_mesh<Flt, 2, 3, mp::device::cpu> mesh{
      mp::view(vertices.data(), mp::make_shape(nvert, 2_s)),
      mp::view(faces.data(), mp::make_shape(nface, 3_s))};
    auto matrix = mp::geometry::build_lumped_mass<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else if (ndim == 3 && dsimplex == 3) {
    mp::geometry::basic_mesh<Flt, 3, 3, mp::device::cpu> mesh{
      mp::view(vertices.data(), mp::make_shape(nvert, 3_s)),
      mp::view(faces.data(), mp::make_shape(nface, 3_s))};
    auto matrix = mp::geometry::build_lumped_mass<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else if (ndim == 3 && dsimplex == 4) {
    mp::geometry::basic_mesh<Flt, 3, 4, mp::device::cpu> mesh{
      mp::view(vertices.data(), mp::make_shape(nvert, 3_s)),
      mp::view(faces.data(), mp::make_shape(nface, 4_s))};
    auto matrix = mp::geometry::build_lumped_mass<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else {
    std::ostringstream oss;
    oss << "Unsupported dimension or simplex type: ndim=" << ndim << ", dsimplex=" << dsimplex;
    throw std::runtime_error(oss.str());
  }
}

void bind_geometry(pybind11::module_& m) {
  m                                        //
      .def("laplacian", laplacian<float>,  //
           "Compute the Laplacian matrix of a mesh.", py::arg("vertices").noconvert(), py::arg("faces").noconvert())
      .def("laplacian", laplacian<double>,  //
           "Compute the Laplacian matrix of a mesh.", py::arg("vertices").noconvert(), py::arg("faces").noconvert())
      .def("lumped_mass", lumped_mass<float>,  //
           "Compute the Lumped-mass matrix of a mesh.", py::arg("vertices").noconvert(), py::arg("faces").noconvert())
      .def("lumped_mass", lumped_mass<double>,  //
           "Compute the Lumped-mass matrix of a mesh.", py::arg("vertices").noconvert(), py::arg("faces").noconvert());
}