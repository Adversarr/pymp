#include "checkings.hpp"


#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

void add_1(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
  auto cnt = arr.size();
  auto ptr = arr.mutable_data();
  for (size_t i = 0; i < cnt; ++i) {
    ptr[i] += 1;
  }
}

void bind_checkings(pybind11::module_& m) {
  m.def("add_1", add_1, "Add 1 to each element of the input array.", py::arg("arr"));
}