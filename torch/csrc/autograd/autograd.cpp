#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/variable.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/ones_like.h>
#endif

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>

#include <c10/util/irange.h>

/*
#ifdef __cilksan__
#ifdef __cplusplus
extern "C" {
#endif
void __csan_default_libhook(uint64_t call_id, uint64_t func_id, unsigned count);
void __csan_free(uint64_t call_id, uint64_t func_id, unsigned count) {
  __csan_default_libhook(call_id, func_id, count);
}
#ifdef __cplusplus
}
#endif
#endif
*/

namespace torch {
namespace autograd {

static void ryan_error_grad() {
    std::cout << "ryan error grad" << std::endl;
}

// NB: This code duplicates existing logic at torch/autograd/__init__.py and
// torch._C._EngineBase.run_backward in torch/csrc/autograd/python_engine.cpp
// This is a purely C++ API for Autograd without any dependencies on python
// it can be exposed in PyTorch C++ API and TorchScript. We will need to
// maintain the logic equality of this file and the python file together if one
// changes.
// TODO: Make the Python API above to just call this C++ API.
variable_list _make_grads(
    const variable_list& outputs,
    const variable_list& grad_outputs) {
  size_t num_tensors = outputs.size();
  size_t num_gradients = grad_outputs.size();
  variable_list new_grads;
  new_grads.reserve(num_tensors);
  if (grad_outputs.empty()) {
    for (const Variable& output : outputs) {
      if (output.requires_grad()) {
        TORCH_CHECK(
            output.numel() == 1,
            "grad can be implicitly created only for scalar outputs");
        new_grads.emplace_back(
            at::ones_like(output, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
      }
    }
  } else {
    TORCH_CHECK(
        num_tensors == num_gradients,
        "got ",
        num_tensors,
        " tensors and ",
        num_gradients,
        " gradients");
    for (const auto i : c10::irange(outputs.size())) {
      const Variable& output = outputs[i];
      const Variable& grad_output = grad_outputs[i];
      if (!grad_output.defined()) {
        if (output.requires_grad()) {
          TORCH_CHECK(
              output.numel() == 1,
              "grad can be implicitly created only for scalar outputs");
          new_grads.emplace_back(
              at::ones_like(output, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
        }
      } else {
        TORCH_CHECK(
            grad_output.is_complex() == output.is_complex(),
            "For complex Tensors, both grad_output and output are required ",
            "to have the same dtype. Mismatch in dtype: grad_output[",
            grad_output,
            "] has a dtype of ",
            grad_output.scalar_type(),
            " and output[",
            output,
            "] has a dtype of ",
            output.scalar_type(),
            ".");
        // grad output is defined, just append to the new_grads
        new_grads.emplace_back(grad_output);
      }
    }
  }
  return new_grads;
}
variable_list run_backward(
    const variable_list& outputs,
    const variable_list& grad_outputs,
    bool keep_graph,
    bool create_graph,
    const variable_list& inputs,
    bool allow_unused,
    bool accumulate_grad) {
  size_t num_tensors = outputs.size();
  edge_list roots;
  roots.reserve(num_tensors);
  for (const auto i : c10::irange(num_tensors)) {
    const Variable& output = outputs[i];
    auto gradient_edge = impl::gradient_edge(output);
    /*
    if (!gradient_edge.function) {
        ryan_error_grad();
        for (int i = 0; i < 10; i++) {
            std::cout << "EXCEPTION on cilk worker: " << __cilkrts_get_worker_number() << " grad mode: " << GradMode::is_enabled() << std::endl;
        }
        std::cout << "Output size: " << output.sizes() << std::endl;
        std::cout << "Output tensor: " << output << std::endl;
    } else {
        std::cout << "Gradient edge function: " << gradient_edge.function->name() << " for cilk worker: " << __cilkrts_get_worker_number() << std::endl;
    }
    */
    TORCH_CHECK(
        gradient_edge.function,
        "element ",
        i,
        " of tensors does not require grad and does not have a grad_fn");
    roots.push_back(std::move(gradient_edge));
  }

  edge_list output_edges;
  if (!inputs.empty()) {
    size_t num_inputs = inputs.size();
    output_edges.reserve(num_inputs);
    for (const auto i : c10::irange(num_inputs)) {
      const Variable& input = inputs[i];
      const auto output_nr = input.output_nr();
      auto grad_fn = input.grad_fn();
      if (!grad_fn) {
        grad_fn = impl::try_get_grad_accumulator(input);
      }
      if (accumulate_grad) {
        input.retain_grad();
      }
      TORCH_CHECK(
          input.requires_grad(),
          "One of the differentiated Tensors does not require grad");
      if (!grad_fn) {
        // See NOTE [ Autograd Unreachable Input ] for details
        output_edges.emplace_back(std::make_shared<Identity>(), 0);
      } else {
        output_edges.emplace_back(grad_fn, output_nr);
      }
    }
  }

  variable_list grad_inputs = Engine::get_default_engine().execute(
      roots,
      grad_outputs,
      keep_graph,
      create_graph,
      accumulate_grad,
      output_edges);
  // check if grad_inputs contains None or not base on the allow_unused flag
  if (!inputs.empty() && !allow_unused) {
    size_t num_inputs = inputs.size();
    for (const auto i : c10::irange(num_inputs)) {
      if (!grad_inputs[i].defined()) {
          for (int i = 0; i < 10; i++) {
            std::cout << "EXCEPTION for cilk worker: " << __cilkrts_get_worker_number() << " for tensor: " << grad_inputs[i] << " sizes: " << grad_inputs[i].sizes() << " grad mode: " << GradMode::is_enabled() << std::endl;
          }
      }
      TORCH_CHECK(
          grad_inputs[i].defined(),
          "One of the "
          "differentiated Tensors appears to not have been used "
          "in the graph. Set allow_unused=True if this is the "
          "desired behavior.");
    }
  }
  return grad_inputs;
}

void backward(
    const variable_list& tensors,
    const variable_list& grad_tensors,
    c10::optional<bool> retain_graph,
    bool create_graph,
    const variable_list& inputs) {
  variable_list gradients = _make_grads(tensors, grad_tensors);
  if (!retain_graph) {
    retain_graph = create_graph;
  }
  run_backward(
      tensors,
      gradients,
      retain_graph.value(),
      create_graph,
      inputs,
      /*allow_unused=*/true,
      /*accumulate_grad=*/true);
}

variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs,
    c10::optional<bool> retain_graph,
    bool create_graph,
    bool allow_unused) {
  variable_list gradients = _make_grads(outputs, grad_outputs);
  if (!retain_graph) {
    retain_graph = create_graph;
  }
  return run_backward(
      outputs,
      gradients,
      retain_graph.value(),
      create_graph,
      inputs,
      allow_unused,
      /*accumulate_grad=*/false);
}

namespace forward_ad {

uint64_t enter_dual_level() {
  return ForwardADLevel::get_next_idx();
}

void exit_dual_level(uint64_t level) {
  ForwardADLevel::release_idx(level);
}

} // namespace forward_ad

} // namespace autograd
} // namespace torch
