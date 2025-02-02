#pragma once

#include <atomic>
#include <cstddef>
#include <exception>
#include <iostream>
#include <csignal>
#include <ATen/core/NamedTensor.h>
#include <c10/core/GradMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

#ifdef _OPENMP
#define INTRA_OP_PARALLEL

#include <omp.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cstdlib>
#endif

namespace at {

#ifdef _OPENMP
namespace internal {

template <typename F>
inline void invoke_parallel(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const F& f) {
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;

  /*
  int64_t num_threads = __cilkrts_get_nworkers();
  // sometimes (during testing?) grain size can be 0
  if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
  }

  int64_t chunk_size = divup((end - begin), num_threads);

  for(int64_t i = begin; i < end; i += chunk_size) {
      f(i, std::min(end, i + chunk_size));
  }
  */
  int64_t num_threads = __cilkrts_get_nworkers();
  // sometimes (during testing?) grain size can be 0
  if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
  }

  int64_t chunk_size = divup((end - begin), num_threads);
  /*
  if (grain_size == 0) {
      grain_size = 1;
  }
  */

  bool tmp_grad_mode = GradMode::is_enabled();
  auto tmp_key_set = c10::impl::tls_local_dispatch_key_set();
  bool prev_names_mode = NamesMode::is_enabled();
  // cilk_for (int64_t i = begin; i < end; i += grain_size) {
  cilk_for (int64_t i = begin; i < end; i += chunk_size) {
      GradMode::set_enabled(tmp_grad_mode);
      _force_tls_local_dispatch_key_set(tmp_key_set);
      NamesMode::set_enabled(prev_names_mode);
      // f(i, std::min(end, i + grain_size));
      f(i, std::min(end, i + chunk_size));
  }

  GradMode::set_enabled(tmp_grad_mode);
  _force_tls_local_dispatch_key_set(tmp_key_set);
  NamesMode::set_enabled(prev_names_mode);

/*
#pragma omp parallel
  {
    // choose number of tasks based on grain size and number of threads
    // can't use num_threads clause due to bugs in GOMP's thread pool (See
    // #32008)
    int64_t num_threads = omp_get_num_threads();
    if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      try {
        internal::ThreadIdGuard tid_guard(tid);
        f(begin_tid, std::min(end, chunk_size + begin_tid));
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
*/
}
} // namespace internal
#endif // _OPENMP

} // namespace at
