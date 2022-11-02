#pragma once

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <new>
#include <functional>
#include <cstdlib>
#include <c10/util/Exception.h>
#include <ATen/core/NamedTensor.h>
#include <c10/core/GradMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

#include <functional>
#include <memory>
#include <iostream>
#include <typeinfo>

// code from online start
template<int Ind, typename R, typename... Args>
struct Wrapper;

template <int Ind, typename Callable, typename R, typename... Args>
R wrapper(Args... args);

template<int Ind, typename Callable>
struct CallableStorage
{
    static void store(Callable callable_)
    {
        callable = std::make_unique<Callable>(callable_);
    }

    template <typename R, typename... Args>
    static R invoke(Args... args)
    {
        return (*callable)(args...);
    }

private:
    static std::unique_ptr<Callable> callable;
};

template <int Ind, typename Callable>
std::unique_ptr<Callable> CallableStorage<Ind, Callable>::callable;

template<int Ind, typename R, typename... Args>
struct Wrapper<Ind, R(Args...)>
{
    using RawType = R(*)(Args...);

    template<typename Callable>
    static RawType wrap(Callable callable)
    {
        CallableStorage<Ind, Callable>::store(callable);
        return wrapper<Ind, Callable, R, Args...>;
    }
};

template <int Ind, typename Callable, typename R, typename... Args>
R wrapper(Args... args)
{
    return CallableStorage<Ind, Callable>::template invoke<R>(args...);
}

// code from online end

template <class scalar_t>
void identity_(scalar_t ident, void* v) {
    *static_cast<scalar_t*>(v) = ident;
}

template <class scalar_t, class SF>
void reduce_(const SF& sf, void* left, void* right) {
    scalar_t tmp_left = *(static_cast<scalar_t*>(left));
    scalar_t tmp_right = *(static_cast<scalar_t*>(right));
    *(static_cast<scalar_t*>(left)) = sf(tmp_left, tmp_right);
}

template <class scalar_t, class SF, class F>
class ParallelReducer
{
public:
  ParallelReducer(scalar_t ident, const SF& sf, const F& f, int64_t grain_size): ident(ident), sf(sf), f(f), grain_size_(grain_size) {}

  scalar_t reduce(int64_t begin, int64_t end) {
      std::cout << "Parallel reduce" << std::endl;
      auto ident_func = std::bind(identity_<scalar_t>, ident, std::placeholders::_1);
      auto reduce_func = std::bind(reduce_<scalar_t, SF>, sf, std::placeholders::_1, std::placeholders::_2);
      auto wrapped_ident_func = Wrapper<0, void(void*)>::wrap(ident_func);
      auto wrapped_reduce_func = Wrapper<1, void(void*, void*)>::wrap(reduce_func);
      scalar_t cilk_reducer(wrapped_ident_func, wrapped_reduce_func) res = ident;

      bool tmp_grad_mode = at::GradMode::is_enabled();
      auto tmp_key_set = c10::impl::tls_local_dispatch_key_set();
      bool prev_names_mode = at::NamesMode::is_enabled();

      cilk_for (int64_t i = begin; i < end; i += grain_size_) {
          at::GradMode::set_enabled(tmp_grad_mode);
          _force_tls_local_dispatch_key_set(tmp_key_set);
          at::NamesMode::set_enabled(prev_names_mode);
          scalar_t tmp = f(i, std::min(end, i + grain_size_), ident);
          scalar_t res_tmp = *&res;
          scalar_t res_tmp_2 = sf(res_tmp, tmp);
          *&res = res_tmp_2;
      }
      at::GradMode::set_enabled(tmp_grad_mode);
      _force_tls_local_dispatch_key_set(tmp_key_set);
      at::NamesMode::set_enabled(prev_names_mode);
      return *&res;
  }
  
public:
  scalar_t ident;
  const SF& sf;
  const F& f;
  int64_t grain_size_;
};

