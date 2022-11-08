#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/CPUBlas.h>
#include <c10/util/irange.h>
#include <c10/core/GradMode.h>
#include <ATen/core/NamedTensor.h>
#include <c10/core/GradMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <cilk/cilk.h>
#include <iostream>

namespace at {
namespace native {
namespace cpublas {
namespace {

template <typename scalar_t, typename opmath_t>
void scale_(int64_t m, int64_t n, opmath_t alpha, scalar_t *a, int64_t lda) {
  if (alpha == opmath_t(1)) {
    return;  // identity
  }

  if (alpha == opmath_t(0)) {
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        a[j * lda + i] = scalar_t(0);
      }
    }
    return;
  }

  for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
      a[j * lda + i] *= alpha;
    }
  }
}


template <typename scalar_t, typename opmath_t>
void gemm_notrans_(
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b)
  for (const auto l : c10::irange(k)) {
    for (const auto j : c10::irange(n)) {
      opmath_t val = b[l + j * ldb] * alpha;
      int64_t i_m = m / 4;
      for (const auto i_i : c10::irange(i_m)) {
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      int64_t i = i_m * 4;
      for (; i < m; i++)
        c[j * ldc + i] += a[i + l * lda] * val;
    }
  }
}

template <typename scalar_t, typename opmath_t>
void gemm_transa_(
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
  const scalar_t *a_ = a;
  for (const auto i : c10::irange(m)) {
    const scalar_t *b_ = b;
    for (const auto j : c10::irange(n)) {
      opmath_t sum = 0;
      for (const auto l : c10::irange(k)) {
        sum += static_cast<opmath_t>(a_[l]) * static_cast<opmath_t>(b_[l]);
      }
      b_ += ldb;
      if (beta == scalar_t(0))
        c[j*ldc+i] = alpha*sum;
      else
        c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
    }
    a_ += lda;
  }
}

template <typename scalar_t, typename opmath_t>
void gemm_transb_(
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b.T)
  for (const auto l : c10::irange(k)) {
    for (const auto j : c10::irange(n)) {
      opmath_t val = b[j + l * ldb] * alpha;
      int64_t i_m = m / 4;
      for (const auto i_i : c10::irange(i_m)) {
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      int64_t i = i_m * 4;
      for (; i < m; i++)
        c[j * ldc + i] += a[i + l * lda] * val;
    }
  }
}

template <typename scalar_t, typename opmath_t>
void gemm_transab_(
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a.T @ b.T)
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(n)) {
      int64_t l_k = k / 4;
      for (const auto l_l : c10::irange(l_k)) {
        c[j * ldc + i] += a[i * lda + l_l * 4 + 0] //
            * (b[(l_l * 4 + 0) * ldb + j] * alpha);
        c[j * ldc + i] += a[i * lda + l_l * 4 + 1] //
            * (b[(l_l * 4 + 1) * ldb + j] * alpha);
        c[j * ldc + i] += a[i * lda + l_l * 4 + 2] //
            * (b[(l_l * 4 + 2) * ldb + j] * alpha);
        c[j * ldc + i] += a[i * lda + l_l * 4 + 3] //
            * (b[(l_l * 4 + 3) * ldb + j] * alpha);
      }
      int64_t l = l_k * 4;
      for (; l < k; l++)
        c[j * ldc + i] += a[i * lda + l] * (b[l * ldb + j] * alpha);
    }
  }
}

// TODO: Begin cilk matmul code
const int64_t BASE = 32768;

template <typename F>
__attribute__((always_inline))
static void buffer_init(F *__restrict__ dst, const F *__restrict__ src,
                        int64_t m, int64_t n, int64_t mstride, int64_t nstride,
                        bool transpose, bool flip) {
  if (!flip) {
    if (!transpose) {
      for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
          dst[j * m + i] = src[j * mstride + i];
    } else {
      // TODO: Seems to be a simple sequential copy, similar to the case above
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          dst[i * n + j] = src[i * nstride + j];
        }
      }
    }
  } else {
    if (!transpose) {
      for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
          dst[i * n + j] = src[j * mstride + i];
    } else {
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          dst[j * m + i] = src[i * nstride + j];
        }
      }
    }
  }
}

#define ARG_INDEX(arg, ii, m, jj, n, transpose)         \
  ((transpose) ? arg[((jj) * m) + (ii)] : arg[((ii) * n) + (jj)])

// A simple and general vectorized base case for matrix multiply.
// This base case computes a INum x JNum submatrix in column-major
// order from a INum subcolumn of A and a JNum subrow of B.
template <typename F, int64_t INum, int64_t JNum, bool transpose_lhs, bool transpose_rhs>
__attribute__((always_inline))
void matmul_vec
(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
 int64_t i, int64_t j, int64_t l,
 int64_t lda, int64_t ldb, int64_t ldc) noexcept {
  // Vector type
  static_assert(std::is_same<F, float>::value || std::is_same<F, double>::value);
  typedef F vF __attribute__((vector_size(sizeof(F)*INum)));
  vF outv[JNum];

  // Zero-initialize output vectors.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum)
    outv[vnum] = (vF){ 0.0 };

  // Get INum values from a column of lhs.
  vF lhsv;
#pragma clang loop unroll(full)
  for (int64_t vidx = 0; vidx < INum; ++vidx) {
    // lhsv[vidx] = ARG_INDEX(lhs, l, lda, i+vidx, lda, transpose_lhs);
    // lhsv[vidx] = ARG_INDEX(lhs, l, kstride, i+vidx, mstride, transpose_lhs);
    lhsv[vidx] = ARG_INDEX(lhs, l, ldc, i+vidx, lda, transpose_lhs);
  }

  // Fill each rhs vector with a value from one of INum rows of rhs.
  vF rhsv[JNum];
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
    // Read the value from a row of rhs.
    // F rhs_val = ARG_INDEX(rhs, j+vnum, nstride, l, kstride, transpose_rhs);
    // F rhs_val = ARG_INDEX(rhs, j+vnum, ldb, l, ldb, transpose_rhs);
    F rhs_val = ARG_INDEX(rhs, j+vnum, ldb, l, ldc, transpose_rhs);
    // Broadcast that value through one of the rhsv.
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < INum; ++vidx)
      rhsv[vnum][vidx] = rhs_val;
  }

  // Each output vector gets the element-wise product of lhsv and one
  // of the rhsv.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum)
    outv[vnum] = lhsv * rhsv[vnum];

  // Add the output vectors to the output matrix.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < JNum; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < INum; ++vidx) {
      // out[(j+vnum) * mstride + (i+vidx)] += outv[vnum][vidx];
      // out[(j+vnum) * ldc + (i+vidx)] += outv[vnum][vidx];
      out[(j+vnum) * lda + (i+vidx)] += outv[vnum][vidx];
    }
  }
}

// A specialized base case that computes the outer product of
// subcolumns of A and subrows of B.  Unlike the more general
// vectorized base case, this version uses fewer memory accesses by
// storing the outer-product result in vector registers.
template <typename F, int64_t KNum>
__attribute__((always_inline))
void matmul_vec_op
(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
 int64_t i, int64_t j, int64_t l,
 int64_t lda, int64_t ldb, int64_t ldc) noexcept {

  // Vector type
  typedef F vF __attribute__((vector_size(sizeof(F)*8)));

  // Vectors storing output submatrix.
  vF outv[4];

  // Zero-initialize the output vectors.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < 4; ++vnum)
    outv[vnum] = (vF){ 0.0 };

  for (int64_t my_l = l; my_l < l + KNum; ++my_l) {
    // Store a subcolumn of lhs into lhsv.
    vF lhsv;
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < 8; ++vidx)
      // lhsv[vidx] = ARG_INDEX(lhs, my_l, lda, i+vidx, lda, false);
      lhsv[vidx] = ARG_INDEX(lhs, my_l, ldc, i+vidx, lda, false);
      // lhsv[vidx] = ARG_INDEX(lhs, my_l, kstride, i+vidx, mstride, false);

    // Store a subrow of rhs into rhsv, replicated twice.
    vF rhsv;
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < 4; ++vidx) {
      // rhsv[vidx] = ARG_INDEX(rhs, j+vidx, ldb, my_l, ldb, true);
      rhsv[vidx] = ARG_INDEX(rhs, j+vidx, ldb, my_l, ldc, true);
      // rhsv[vidx] = ARG_INDEX(rhs, j+vidx, nstride, my_l, kstride, true);
      rhsv[vidx + 4] = rhsv[vidx];
    }

    // Perform the multiplications.
    outv[0] += lhsv * rhsv;
    vF rhsv_p0 = __builtin_shufflevector(rhsv, rhsv, 1, 0, 3, 2, 5, 4, 7, 6);
    outv[1] += lhsv * rhsv_p0;
    vF rhsv_p1 = __builtin_shufflevector(rhsv, rhsv, 2, 3, 0, 1, 6, 7, 4, 5);
    outv[2] += lhsv * rhsv_p1;
    vF rhsv_p2 = __builtin_shufflevector(rhsv_p0, rhsv_p0, 2, 3, 0, 1, 6, 7, 4, 5);
    outv[3] += lhsv * rhsv_p2;
  }

  // Shuffle the output vectors to support simple vector-add
  // operations to store the result back into the output matrix.
  vF st[8];
  // A0B0, A1B0, A2B2, A3B2, A4B0, A5B0, A6B2, A7B2
  st[0] = __builtin_shufflevector(outv[0], outv[1], 0, 9, 2, 11, 4, 13, 6, 15);
  // A0B1, A1B1, A2B3, A3B3, A4B1, A5B1, A6B3, A7B3
  st[1] = __builtin_shufflevector(outv[1], outv[0], 0, 9, 2, 11, 4, 13, 6, 15);
  // A0B2, A1B2, A2B0, A3B0, A4B2, A5B2, A6B0, A7B0
  st[2] = __builtin_shufflevector(outv[2], outv[3], 0, 9, 2, 11, 4, 13, 6, 15);
  // A0B3, A1B3, A2B1, A3B1, A4B3, A5B3, A6B1, A7B1
  st[3] = __builtin_shufflevector(outv[3], outv[2], 0, 9, 2, 11, 4, 13, 6, 15);

  // A0B0, A1B0, A2B0, A3B0, A4B0, A5B0, A6B0, A7B0
  st[4] = __builtin_shufflevector(st[0], st[2], 0, 1, 10, 11, 4, 5, 14, 15);
  // A0B1, A1B1, A2B1, A3B1, A4B1, A5B1, A6B1, A7B1
  st[5] = __builtin_shufflevector(st[1], st[3], 0, 1, 10, 11, 4, 5, 14, 15);
  // A0B2, A1B2, A2B2, A3B2, A4B2, A5B2, A6B2, A7B2
  st[6] = __builtin_shufflevector(st[2], st[0], 0, 1, 10, 11, 4, 5, 14, 15);
  // A0B3, A1B3, A2B3, A3B3, A4B3, A5B3, A6B3, A7B3
  st[7] = __builtin_shufflevector(st[3], st[1], 0, 1, 10, 11, 4, 5, 14, 15);


  // Add the output vectors to the output matrix.
#pragma clang loop unroll(full)
  for (int64_t vnum = 0; vnum < 4; ++vnum) {
#pragma clang loop unroll(full)
    for (int64_t vidx = 0; vidx < 8; ++vidx) {
      // out[(j+vnum) * mstride + (i+vidx)] += st[4+vnum][vidx];
      // TODO: here mstride is different from before since the matrices are already packed
      out[(j+vnum) * lda + (i+vidx)] += st[4+vnum][vidx];
    }
  }
}

#ifdef USE_AVX512
const int64_t nVec = 8;
const int64_t mVec = 16;
#else
const int64_t nVec = 4;
const int64_t mVec = 8;
#endif
const int64_t kVec = 16;

template <typename F, typename opmath_t, bool transpose_lhs, bool transpose_rhs, bool small_n = false>
void matmul_base_transa_(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
                 int64_t m, int64_t n, int64_t k,
                 int64_t lda, int64_t ldb, int64_t ldc,
                 opmath_t alpha, opmath_t beta) noexcept {
  // Zero-initialize the temporary buffer for out.
  F outTmp[n*m];
  for (int64_t j = 0; j < n; ++j)
    for (int64_t i = 0; i < m; ++i)
      outTmp[j * m + i] = 0.0;
  F lhsTmp[(k*m)];
  F rhsTmp[(k*n)];
  // buffer_init(lhsTmp, lhs, m, k, mstride, kstride, transpose_lhs, transpose_lhs);
  // buffer_init(rhsTmp, rhs, k, n, kstride, nstride, transpose_rhs, !transpose_rhs);
  buffer_init(lhsTmp, lhs, m, k, lda, ldc, transpose_lhs, transpose_lhs);
  buffer_init(rhsTmp, rhs, k, n, ldc, ldb, transpose_rhs, !transpose_rhs);

  // buffer_init(lhsTmp, lhs, m, k, lda, lda, transpose_lhs, transpose_lhs);
  // buffer_init(rhsTmp, rhs, k, n, ldb, ldb, transpose_rhs, !transpose_rhs);

  for (int64_t jj = 0; jj < n/nVec; ++jj) {
    for (int64_t ii = 0; ii < m/mVec; ++ii) {
      for (int64_t ll = 0; ll < k/kVec; ++ll) {
        matmul_vec_op<F, kVec>
          (outTmp, lhsTmp, rhsTmp, mVec * ii, nVec * jj, kVec * ll, m, n, k);
      }
      for (int64_t l = kVec * (k/kVec); l < k; ++l)
        matmul_vec<F, mVec, nVec, false, true>
          (outTmp, lhsTmp, rhsTmp, mVec * ii, nVec * jj, l, m, n, k);
    }
    for (int64_t j = nVec * jj; j < nVec * (jj+1); ++j) {
      for (int64_t l = 0; l < k; ++l) {
#pragma clang loop vectorize(disable)
        for (int64_t i = mVec * (m/mVec); i < m; ++i) {
          outTmp[j * m + i] +=
            ARG_INDEX(lhsTmp, l, k, i, m, false) *
            ARG_INDEX(rhsTmp, j, n, l, k, true);
        }
      }
    }
  }
  for (int64_t j = nVec * (n/nVec); j < n; ++j) {
    for (int64_t l = 0; l < k; ++l) {
      // #pragma clang loop vectorize(disable)
      for (int64_t i = 0; i < m; ++i)
          outTmp[j * m + i] +=
            ARG_INDEX(lhsTmp, l, k, i, m, false) *
            ARG_INDEX(rhsTmp, j, n, l, k, true);
    }
  }

  // Add the result of this base-case multiplication back into out.
  // special case for transa_, based on the blas kernel implementation
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      // TODO: fix this case with the weird scaling
      // out[j * mstride + i] = out[j * mstride + i] * beta + alpha * outTmp[j * m + i];
      // out[j * ldc + i] = out[j * ldc + i] * beta + alpha * outTmp[j * m + i];
      // out[j * lda + i] = out[j * lda + i] * beta + alpha * outTmp[j * m + i];
      out[j * lda + i] += alpha * outTmp[j * m + i];
    }
  }
}

template <typename F, typename opmath_t, bool transpose_lhs, bool transpose_rhs, bool small_n = false>
void matmul_base(F *__restrict__ out, const F *__restrict__ lhs, const F *__restrict__ rhs,
                 int64_t m, int64_t n, int64_t k,
                 int64_t lda, int64_t ldb, int64_t ldc,
                 opmath_t alpha, opmath_t beta) noexcept {
  if (transpose_lhs && !transpose_rhs) {
    matmul_base_transa_<F, opmath_t, transpose_lhs, transpose_rhs, small_n>
        (out, lhs, rhs, m, n, k, lda, ldb, ldc, alpha, beta);
    return;
  }

  // Zero-initialize the temporary buffer for out.
  F outTmp[n*m];
  for (int64_t j = 0; j < n; ++j)
    for (int64_t i = 0; i < m; ++i)
      outTmp[j * m + i] = 0.0;
  F lhsTmp[(k*m)];
  F rhsTmp[(k*n)];
  // buffer_init(lhsTmp, lhs, m, k, mstride, kstride, transpose_lhs, transpose_lhs);
  // buffer_init(rhsTmp, rhs, k, n, kstride, nstride, transpose_rhs, !transpose_rhs);
  buffer_init(lhsTmp, lhs, m, k, lda, ldc, transpose_lhs, transpose_lhs);
  buffer_init(rhsTmp, rhs, k, n, ldc, ldb, transpose_rhs, !transpose_rhs);

  // buffer_init(lhsTmp, lhs, m, k, lda, k, transpose_lhs, transpose_lhs);
  // buffer_init(rhsTmp, rhs, k, n, k, ldb, transpose_rhs, !transpose_rhs);

  for (int64_t jj = 0; jj < n/nVec; ++jj) {
    for (int64_t ii = 0; ii < m/mVec; ++ii) {
      for (int64_t ll = 0; ll < k/kVec; ++ll) {
        matmul_vec_op<F, kVec>
          (outTmp, lhsTmp, rhsTmp, mVec * ii, nVec * jj, kVec * ll, m, n, k);
      }
      for (int64_t l = kVec * (k/kVec); l < k; ++l)
        matmul_vec<F, mVec, nVec, false, true>
          (outTmp, lhsTmp, rhsTmp, mVec * ii, nVec * jj, l, m, n, k);
    }
    for (int64_t j = nVec * jj; j < nVec * (jj+1); ++j) {
      for (int64_t l = 0; l < k; ++l) {
#pragma clang loop vectorize(disable)
        for (int64_t i = mVec * (m/mVec); i < m; ++i) {
          outTmp[j * m + i] +=
            ARG_INDEX(lhsTmp, l, k, i, m, false) *
            ARG_INDEX(rhsTmp, j, n, l, k, true);
        }
      }
    }
  }
  for (int64_t j = nVec * (n/nVec); j < n; ++j) {
    for (int64_t l = 0; l < k; ++l) {
      // #pragma clang loop vectorize(disable)
      for (int64_t i = 0; i < m; ++i)
          outTmp[j * m + i] +=
            ARG_INDEX(lhsTmp, l, k, i, m, false) *
            ARG_INDEX(rhsTmp, j, n, l, k, true);
    }
  }

  // Add the result of this base-case multiplication back into out.
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      // out[j * mstride + i] += alpha * outTmp[j * m + i];
      // out[j * ldc + i] += alpha * outTmp[j * m + i];
      out[j * lda + i] += alpha * outTmp[j * m + i];
    }
  }
}

__attribute__((always_inline))
static int64_t split_dim(int64_t n) {
  // Special case: n is a power of 2.
  if ((n & -n) == n)
    return n/2;
  const int64_t split = 1 << (64 - __builtin_clzl(n - 1));
  return split / 2;
}

template <typename scalar_t, typename opmath_t>
void matmul_dac(scalar_t *c, const scalar_t *a, const scalar_t *b,
                int64_t m, int64_t n, int64_t k,
                int64_t lda, int64_t ldb, int64_t ldc,
                opmath_t alpha, opmath_t beta,
                bool transpose_lhs, bool transpose_rhs, bool grad_mode, c10::impl::LocalDispatchKeySet keyset, bool names_mode) noexcept {
  GradMode::set_enabled(grad_mode);
  _force_tls_local_dispatch_key_set(keyset);
  NamesMode::set_enabled(names_mode);

  if (m == 0 || n == 0 || k == 0) {
    return;
  }

  if ((m * n) + (m * k) + (n * k) <= BASE / sizeof(scalar_t)) {
    if (transpose_lhs && transpose_rhs) {
        matmul_base<scalar_t, opmath_t, true, true, false>
          (c, a, b, m, n, k, lda, ldb, ldc, alpha, beta);
    } else if (transpose_lhs && !transpose_rhs) {
        matmul_base<scalar_t, opmath_t, true, false, false>
          (c, a, b, m, n, k, lda, ldb, ldc, alpha, beta);
    } else if (!transpose_lhs && transpose_rhs) {
        matmul_base<scalar_t, opmath_t, false, true, false>
          (c, a, b, m, n, k, lda, ldb, ldc, alpha, beta);
    } else {
        matmul_base<scalar_t, opmath_t, false, false, false>
          (c, a, b, m, n, k, lda, ldb, ldc, alpha, beta);
    }
    return;
  }

  // Split the maximum dimension
  const int64_t max_dim = std::max(std::max(m, n), k);
  // We prefer to spawn higher in the recursion tree than lower.
  // Because the base case vectorizes over dimension m, which is the
  // fastest moving dimension of the output matrix, we prefer to split
  // n instead of m.
  if (max_dim == n) {
    const int64_t split = split_dim(n);
    cilk_spawn matmul_dac<scalar_t, opmath_t>
      (c,
       // &ARG_INDEX(a, 0, kstride, 0, mstride, transpose_lhs),
       // &ARG_INDEX(b, 0, nstride, 0, kstride, transpose_rhs),
       // &ARG_INDEX(a, 0, lda, 0, lda, transpose_lhs),
       // &ARG_INDEX(b, 0, ldb, 0, ldb, transpose_rhs),
       &ARG_INDEX(a, 0, ldc, 0, lda, transpose_lhs),
       &ARG_INDEX(b, 0, ldb, 0, ldc, transpose_rhs),
       m, split, k, lda, ldb, ldc,
       alpha, beta,
       transpose_lhs, transpose_rhs,
       grad_mode, keyset, names_mode
       );
    matmul_dac<scalar_t, opmath_t>
      // (c + (split * mstride),
      // (c + (split * ldc),
      (c + (split * lda),
       // &ARG_INDEX(a, 0, kstride, 0, mstride, transpose_lhs),
       // &ARG_INDEX(b, split, nstride, 0, kstride, transpose_rhs),
       // &ARG_INDEX(a, 0, lda, 0, lda, transpose_lhs),
       // &ARG_INDEX(b, split, ldb, 0, ldb, transpose_rhs),
       &ARG_INDEX(a, 0, ldc, 0, lda, transpose_lhs),
       &ARG_INDEX(b, split, ldb, 0, ldc, transpose_rhs),
       m, (n - split), k, lda, ldb, ldc,
       alpha, beta,
       transpose_lhs, transpose_rhs,
       grad_mode, keyset, names_mode
       );
    cilk_sync;
  } else if (max_dim == m) {
    const int64_t split = split_dim(m);
    cilk_spawn matmul_dac<scalar_t, opmath_t>
      (c,
       // &ARG_INDEX(a, 0, kstride, 0, mstride, transpose_lhs),
       // &ARG_INDEX(b, 0, nstride, 0, kstride, transpose_rhs),
       // &ARG_INDEX(a, 0, lda, 0, lda, transpose_lhs),
       // &ARG_INDEX(b, 0, ldb, 0, ldb, transpose_rhs),
       &ARG_INDEX(a, 0, ldc, 0, lda, transpose_lhs),
       &ARG_INDEX(b, 0, ldb, 0, ldc, transpose_rhs),
       split, n, k, lda, ldb, ldc,
       alpha, beta,
       transpose_lhs, transpose_rhs,
       grad_mode, keyset, names_mode
       );
    matmul_dac<scalar_t, opmath_t>
      (c + split,
       // &ARG_INDEX(a, 0, kstride, split, mstride, transpose_lhs),
       // &ARG_INDEX(b, 0, nstride, 0, kstride, transpose_rhs),
       // &ARG_INDEX(a, 0, lda, split, lda, transpose_lhs),
       // &ARG_INDEX(b, 0, ldb, 0, ldb, transpose_rhs),
       &ARG_INDEX(a, 0, ldc, split, lda, transpose_lhs),
       &ARG_INDEX(b, 0, ldb, 0, ldc, transpose_rhs),
       (m - split), n, k, lda, ldb, ldc,
       alpha, beta,
       transpose_lhs, transpose_rhs,
       grad_mode, keyset, names_mode
       );
    cilk_sync;
  } else { // max_dim == k
    const int64_t split = split_dim(k);
    matmul_dac<scalar_t, opmath_t>
      (c,
       // &ARG_INDEX(a, 0, kstride, 0, mstride, transpose_lhs),
       // &ARG_INDEX(b, 0, nstride, 0, kstride, transpose_rhs),
       // &ARG_INDEX(a, 0, lda, 0, lda, transpose_lhs),
       // &ARG_INDEX(b, 0, ldb, 0, ldb, transpose_rhs),
       &ARG_INDEX(a, 0, ldc, 0, lda, transpose_lhs),
       &ARG_INDEX(b, 0, ldb, 0, ldc, transpose_rhs),
       m, n, split, lda, ldb, ldc,
       alpha, beta,
       transpose_lhs, transpose_rhs,
       grad_mode, keyset, names_mode
       );
    matmul_dac<scalar_t, opmath_t>
      (c,
       // &ARG_INDEX(a, split, kstride, 0, mstride, transpose_lhs),
       // &ARG_INDEX(b, 0, nstride, split, kstride, transpose_rhs),
       // &ARG_INDEX(a, split, lda, 0, lda, transpose_lhs),
       // &ARG_INDEX(b, 0, ldb, split, ldb, transpose_rhs),
       &ARG_INDEX(a, split, ldc, 0, lda, transpose_lhs),
       &ARG_INDEX(b, 0, ldb, split, ldc, transpose_rhs),
       m, n, (k - split), lda, ldb, ldc,
       alpha, beta,
       transpose_lhs, transpose_rhs,
       grad_mode, keyset, names_mode
       );
  }
}

template <typename scalar_t, typename opmath_t>
void cilk_matmul(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  static_assert(std::is_same<scalar_t, float>::value || std::is_same<scalar_t, double>::value);
  bool grad_mode = c10::GradMode::is_enabled();
  auto keyset = c10::impl::tls_local_dispatch_key_set();
  bool names_mode = NamesMode::is_enabled();
  scalar_t* tmp_a = new scalar_t[m * k];
  scalar_t* tmp_b = new scalar_t[k * n];
  scalar_t* tmp_c = new scalar_t[m * n];

  if (transa == TransposeType::Transpose) {
      for (const auto j : c10::irange(m)) {
        for (const auto i : c10::irange(k)) {
          // tmp_a[i * m + j] = a[j * lda + i];
          //  tmp_a[i * m + j] = a[j * lda + i];
          // make sure right now tmp_a is still in column-major order
          tmp_a[i * m + j] = a[j * lda + i];
        }
      }
  } else {
  for (const auto j : c10::irange(k)) {
    for (const auto i : c10::irange(m)) {
      tmp_a[j * m + i] = a[j * lda + i];
    }
  }
  }

  if (transb == TransposeType::Transpose) {
      for (const auto j : c10::irange(k)) {
        for (const auto i : c10::irange(n)) {
          tmp_b[i * k + j] = b[j * ldb + i];
        }
      }
  } else {
      for (const auto j : c10::irange(n)) {
        for (const auto i : c10::irange(k)) {
          tmp_b[j * k + i] = b[j * ldb + i];
        }
      }
  }


  for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
      tmp_c[j * m + i] = c[j * ldc + i];
    }
  }
    scale_(m, n, beta, tmp_c, m);
    matmul_dac<scalar_t, opmath_t>(tmp_c, tmp_a, tmp_b, m, n, k, m, n, k, alpha, beta, false, false, grad_mode, keyset, names_mode);

    /*
  if (transa == TransposeType::NoTranspose && transb == TransposeType::NoTranspose) {
    scale_(m, n, beta, tmp_c, m);
    matmul_dac<scalar_t, opmath_t>(tmp_c, tmp_a, tmp_b, m, n, k, m, n, k, alpha, beta, false, false, grad_mode, keyset, names_mode);
    // scale_(m, n, beta, c, ldc);
    // matmul_dac<scalar_t, opmath_t>(c, a, b, m, n, k, lda, ldb, ldc, alpha, beta, false, false, grad_mode, keyset, names_mode);
  } else if (transa == TransposeType::Transpose && transb != TransposeType::Transpose) {
    // matmul_dac<scalar_t, opmath_t>(c, a, b, m, n, k, lda, ldb, ldc, alpha, beta, true, false, grad_mode, keyset, names_mode);
    matmul_dac<scalar_t, opmath_t>(tmp_c, tmp_a, tmp_b, m, n, k, m, n, k, alpha, beta, true, false, grad_mode, keyset, names_mode);
  } else if (transa == TransposeType::NoTranspose && transb == TransposeType::Transpose) {
    // scale_(m, n, beta, c, ldc);
    // matmul_dac<scalar_t, opmath_t>(c, a, b, m, n, k, lda, ldb, ldc, alpha, beta, false, true, grad_mode, keyset, names_mode);
    scale_(m, n, beta, tmp_c, m);
    matmul_dac<scalar_t, opmath_t>(tmp_c, tmp_a, tmp_b, m, n, k, m, n, k, alpha, beta, false, true, grad_mode, keyset, names_mode);
  } else {  // transa == TransposeType::Transpose && transb == TransposeType::Transpose
    // scale_(m, n, beta, c, ldc);
    // matmul_dac<scalar_t, opmath_t>(c, a, b, m, n, k, lda, ldb, ldc, alpha, beta, true, true, grad_mode, keyset, names_mode);
    scale_(m, n, beta, tmp_c, m);
    matmul_dac<scalar_t, opmath_t>(tmp_c, tmp_a, tmp_b, m, n, k, m, n, k, alpha, beta, true, true, grad_mode, keyset, names_mode);
  }
  */

  for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
      c[j * ldc + i] = tmp_c[j * m + i];
    }
  }

  delete[] tmp_a;
  delete[] tmp_b;
  delete[] tmp_c;
}

// end cilk_matmul
template <typename scalar_t, typename opmath_t>
void gemm_core_(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {

    constexpr bool is_float = std::is_same<scalar_t, float>::value;
    constexpr bool is_double = std::is_same<scalar_t, double>::value;
    if constexpr(is_float) {
        // std::cout << "matmul for: " << m << " " << n << " " << k << " " << lda << " " << ldb << " " << ldc << std::endl;
        // testing code to make sure the matmul is correct
        /*
        scalar_t* tmp_c = new scalar_t[ldc * n];
        for (int i = 0; i < ldc * n; i++) {
            tmp_c[i] = std::numeric_limits<float>::infinity();
        }
        for (const auto j : c10::irange(n)) {
          for (const auto i : c10::irange(m)) {
            tmp_c[j * ldc + i] = c[j * ldc + i];
          }
        }
        */
        cilk_matmul<scalar_t, opmath_t>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        /*
          if(transa == TransposeType::NoTranspose && transb == TransposeType::NoTranspose) {
            gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, tmp_c, ldc);
          } else if(transa == TransposeType::Transpose && transb != TransposeType::Transpose) {
            gemm_transa_(m, n, k, alpha, a, lda, b, ldb, beta, tmp_c, ldc);
          } else if(transa == TransposeType::NoTranspose && transb == TransposeType::Transpose) {
            gemm_transb_(m, n, k, alpha, a, lda, b, ldb, beta, tmp_c, ldc);
          } else {  // transa == TransposeType::Transpose && transb == TransposeType::Transpose
            gemm_transab_(m, n, k, alpha, a, lda, b, ldb, beta, tmp_c, ldc);
          }
        for (const auto j : c10::irange(n)) {
          for (const auto i : c10::irange(m)) {
            bool b1 = (std::abs(static_cast<float>(tmp_c[j * ldc + i]) - static_cast<float>(c[j * ldc + i])) < 1e-5);
            if (!b1 && transa == TransposeType::Transpose && transb == TransposeType::Transpose) {
                std::cout << "M: " << m << " n: " << n << " k: " << k << " lda: " << lda << " ldb: " << ldb << " ldc: " << ldc << " alpha: " << alpha << " beta: " << beta << std::endl;
                std::cout << "Error for row: " << i << " column: " << j << " what cilk got: " << c[j * ldc + i] << " what libtorch got: " << tmp_c[j * ldc + i] << std::endl;
                std::cout << "Transpose type: " << (transa == TransposeType::Transpose) << " " << (transb == TransposeType::Transpose) << std::endl;
            } else if (!b1) {
                std::cout << "M: " << m << " n: " << n << " k: " << k << " lda: " << lda << " ldb: " << ldb << " ldc: " << ldc << " alpha: " << alpha << " beta: " << beta << std::endl;
                std::cout << "Error for row: " << i << " column: " << j << " what cilk got: " << c[j * ldc + i] << " what libtorch got: " << tmp_c[j * ldc + i] << std::endl;
                std::cout << "Transpose type: " << (transa == TransposeType::Transpose) << " " << (transb == TransposeType::Transpose) << std::endl;
                TORCH_INTERNAL_ASSERT(false);
            }
          }
        }
        std::cout << "worked out float?" << std::endl;
        delete[] tmp_c;
        */
        return;
    } 
    if constexpr(is_double) {
        /*
        scalar_t* tmp_c = new scalar_t[ldc * n];
        for (const auto j : c10::irange(n)) {
          for (const auto i : c10::irange(m)) {
            tmp_c[j * ldc + i] = c[j * ldc + i];
          }
        }
        */
        cilk_matmul<scalar_t, opmath_t>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        /*
          if(transa == TransposeType::NoTranspose && transb == TransposeType::NoTranspose) {
            gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, tmp_c, ldc);
          } else if(transa == TransposeType::Transpose && transb != TransposeType::Transpose) {
            gemm_transa_(m, n, k, alpha, a, lda, b, ldb, beta, tmp_c, ldc);
          } else if(transa == TransposeType::NoTranspose && transb == TransposeType::Transpose) {
            gemm_transb_(m, n, k, alpha, a, lda, b, ldb, beta, tmp_c, ldc);
          } else {  // transa == TransposeType::Transpose && transb == TransposeType::Transpose
            gemm_transab_(m, n, k, alpha, a, lda, b, ldb, beta, tmp_c, ldc);
          }
        for (const auto j : c10::irange(n)) {
          for (const auto i : c10::irange(m)) {
            bool b1 = (std::abs(static_cast<float>(tmp_c[j * ldc + i]) - static_cast<float>(c[j * ldc + i])) < 1e-8);
            if (!b1) {
                std::cout << "M: " << m << " n: " << n << " k: " << k << " lda: " << lda << " ldb: " << ldb << " ldc: " << ldc << " alpha: " << alpha << " beta: " << beta << std::endl;
                std::cout << "Error for row: " << i << " column: " << j << " what cilk got: " << c[j * ldc + i] << " what libtorch got: " << tmp_c[j * ldc + i] << std::endl;
                if (i * ldc + j < ldc * n) {
                    std::cout << "Val for the opposite: " << tmp_c[i * ldc + j] << std::endl;
                }
                TORCH_INTERNAL_ASSERT(false);
            }
          }
        }
        delete[] tmp_c;
        */
        return;
    } 

  if(transa == TransposeType::NoTranspose && transb == TransposeType::NoTranspose) {
    gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if(transa == TransposeType::Transpose && transb != TransposeType::Transpose) {
    gemm_transa_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if(transa == TransposeType::NoTranspose && transb == TransposeType::Transpose) {
    gemm_transb_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else {  // transa == TransposeType::Transpose && transb == TransposeType::Transpose
    gemm_transab_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}

void cpublas_gemm_impl(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::kHalf, at::kBFloat16,
    type, "cpublas_gemm_impl",
      [&]{
        using opmath_t = at::opmath_type<scalar_t>;
        gemm_core_(
            transa, transb, m, n, k,
            alpha.to<opmath_t>(),
            static_cast<const scalar_t *>(a), lda,
            static_cast<const scalar_t *>(b), ldb,
            beta.to<opmath_t>(),
            static_cast<scalar_t *>(c), ldc);
      });
}

void cpublas_axpy_impl(at::ScalarType type, int64_t n, const Scalar& _a, const void *_x, int64_t incx, void *_y, int64_t incy){
  if (type == at::kBool) {
      auto a = _a.to<bool>();
      auto x = static_cast<const bool *>(_x);
      auto y = static_cast<bool *>(_y);
      int64_t i;
      for(i = 0; i < n; i++)
        y[i*incy] |= a & x[i*incx];
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::kHalf, at::kBFloat16, type, "cpublas_axpy_impl",
      [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        auto a = _a.to<opmath_t>();
        auto x = static_cast<const scalar_t *>(_x);
        auto y = static_cast<scalar_t *>(_y);
        int64_t i;
        for(i = 0; i < n; i++)
          y[i*incy] += a*x[i*incx];
      });
  }
}

void cpublas_copy_impl(at::ScalarType type, int64_t n, const void *_x, int64_t incx, void *_y, int64_t incy){
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(at::kComplexHalf, at::kHalf, at::kBFloat16, at::kBool, type, "cpublas_copy_impl",
    [&] {
      auto x = static_cast<const scalar_t *>(_x);
      auto y = static_cast<scalar_t *>(_y);
      int64_t i;
      for(i = 0; i < n; i++)
        y[i*incy] = x[i*incx];
    });
}

}}  // namespace cpublas::(anonymous)


REGISTER_DISPATCH(cpublas::gemm_stub, &cpublas::cpublas_gemm_impl);
REGISTER_DISPATCH(cpublas::axpy_stub, &cpublas::cpublas_axpy_impl);
REGISTER_DISPATCH(cpublas::copy_stub, &cpublas::cpublas_copy_impl);

}}  // namespace at::native
