#pragma once
#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::eltwise {
namespace py = pybind11;

void bind_add_operation(py::module &m);

}