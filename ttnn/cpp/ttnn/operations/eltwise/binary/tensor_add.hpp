
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/add_device_operation.hpp"

namespace ttnn::operations::binary {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
struct AddOperation {
    // The user will be able to call this method as `Tensor output = ttnn::composite_example(input_tensor)` after the op
    // is registered
    static Tensor invoke(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
        auto result = prim::tensor_add(input_tensor_a, input_tensor_b);
        return result;
    }
};

}  // namespace ttnn::operations::binary

namespace ttnn {
constexpr auto tensor_add = ttnn::register_operation<"ttnn::tensor_add", operations::binary::AddOperation>();
}  // namespace ttnn
