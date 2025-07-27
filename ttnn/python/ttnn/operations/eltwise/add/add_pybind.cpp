#include "add_pybind.hpp"
#include "ttnn/prim/add.hpp"

namespace ttnn::operations::eltwise {
void bind_add_operation(py::module &m) {
    bind_registered_operation(
        m,
        ttnn::prim::add_device,
        R"doc(add(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor)doc",
        ttnn::pybind_overload_t{
            [](decltype(ttnn::prim::add_device) op,
               const ttnn::Tensor &a,
               const ttnn::Tensor &b) {
                return op(a, b);
            },
            py::arg("a"), py::arg("b")
        }
    );
}

}