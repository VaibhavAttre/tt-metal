// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_device_operation.hpp"

namespace ttnn::operations::binary {

AddDeviceOperation::program_factory_t AddDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // bool some_condition_based_on_operation_attributes_and_or_tensor_args = true;
    // if (some_condition_based_on_operation_attributes_and_or_tensor_args) {
    return SingleCore{};
    // }
    // return MultiCore{};
}

// Input arguments validation
void AddDeviceOperation::validate(const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // TODO: The shape of the tensors must be identical
    // For now let's skip the validation.
}

void AddDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate(attributes, tensor_args);
}

void AddDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate(attributes, tensor_args);
}

AddDeviceOperation::spec_return_value_t AddDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    MemoryConfig memory_config{TensorMemoryLayout::INTERLEAVED, BufferType::L1};
    return TensorSpec(
        input_tensor_a.get_logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor_a.get_dtype(), tt::tt_metal::PageConfig(input_tensor_a.get_layout()), memory_config));
}

AddDeviceOperation::tensor_return_value_t AddDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
}

std::tuple<AddDeviceOperation::operation_attributes_t, AddDeviceOperation::tensor_args_t> AddDeviceOperation::invoke(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    return {operation_attributes_t{true, 42}, tensor_args_t{input_tensor_a, input_tensor_b}};
}

}  // namespace ttnn::operations::binary
