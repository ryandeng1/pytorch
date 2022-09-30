#pragma once

#include <c10/macros/Macros.h>

#include <cstdint>
#include <iostream>
#include <cilk/cilk_api.h>

namespace c10 {

static void ryan5() {
}

// Structure used to pack all the thread local boolean
// flags used by autograd
struct C10_API AutogradState {
  static AutogradState& get_tls_state();
  static void set_tls_state(AutogradState state);

  AutogradState(bool grad_mode, bool inference_mode, bool fw_grad_mode)
      : grad_mode_(grad_mode),
        inference_mode_(inference_mode),
        fw_grad_mode_(fw_grad_mode) {
            grad_mode_ = 1;
            inference_mode = 0;
            fw_grad_mode = 1;
        }
  


  void set_grad_mode(bool enabled) {
    if (!enabled) {
        if (__cilkrts_get_worker_number() != 0) {
            ryan5();
        }
        return;
    } else {
        std::cout << "set grad_mode: " << enabled << " by cilk worker: " << __cilkrts_get_worker_number() << std::endl;
        if (__cilkrts_get_worker_number() != 0) {
            ryan5();
        }
    }
    grad_mode_ = enabled;
  }

  void set_fw_grad_mode(bool enabled) {
    if (!enabled) {
        if (__cilkrts_get_worker_number() != 0) {
            ryan5();
        }
        return;
    } else {
        std::cout << "set fw_grad_mode: " << enabled << " by cilk worker: " << __cilkrts_get_worker_number() << std::endl;
    }
    fw_grad_mode_ = enabled;
  }

  void set_inference_mode(bool enabled) {
    if (enabled) {
        return;
    } else {
        std::cout << "set inference mode: " << enabled << " by cilk worker: " << __cilkrts_get_worker_number() << std::endl;
    }
    inference_mode_ = enabled;
  }

  bool get_grad_mode() const {
    return grad_mode_;
  }

  bool get_fw_grad_mode() const {
    return fw_grad_mode_;
  }

  bool get_inference_mode() const {
    return inference_mode_;
  }

 private:
  bool grad_mode_ : 1;
  bool inference_mode_ : 1;
  bool fw_grad_mode_ : 1;
};

} // namespace c10
