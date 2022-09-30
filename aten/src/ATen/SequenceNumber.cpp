#include <ATen/SequenceNumber.h>
#include <ATen/Parallel.h>
#include <cilk/cilk_api.h>
#include <iostream>
#include <mutex>

namespace at {
namespace sequence_number {

namespace {
// thread_local uint64_t sequence_nr_ = 0;
uint64_t sequence_nr_ = 0;
std::mutex sequence_nr_mutex;
} // namespace

uint64_t peek() {
  std::lock_guard<std::mutex> guard(sequence_nr_mutex);
  return sequence_nr_;
}

uint64_t get_and_increment() {
  std::lock_guard<std::mutex> guard(sequence_nr_mutex);
  // std::cout << "Sequence number get_and_increment by: " << __cilkrts_get_worker_number() << " sn: " << sequence_nr_ <<  std::endl;
  return sequence_nr_++;
}

} // namespace sequence_number
} // namespace at
