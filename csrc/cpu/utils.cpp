#ifndef VLLM_NUMA_DISABLED
  #include <numa.h>
  #include <unistd.h>
  #include <string>
  #include <sched.h>
#endif
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
  #include <unistd.h>
  #include <sys/syscall.h>
  #define gettid() syscall(SYS_gettid)
#endif

#include "cpu/utils.hpp"

#ifdef VLLM_NUMA_DISABLED
std::string init_cpu_threads_env(const std::string& cpu_ids) {
  return std::string(
      "Warning: NUMA is not enabled in this build. `init_cpu_threads_env` has "
      "no effect to setup thread affinity.");
}

#endif

namespace cpu_utils {
ScratchPadManager::ScratchPadManager() : size_(0), ptr_(nullptr) {
  this->realloc(allocation_unit * 128);
}

void ScratchPadManager::realloc(size_t new_size) {
  new_size = round(new_size);
  if (new_size > size_) {
    if (ptr_ != nullptr) {
      std::free(ptr_);
    }
    ptr_ = std::aligned_alloc(64, new_size);
    size_ = new_size;
  }
}

ScratchPadManager* ScratchPadManager::get_scratchpad_manager() {
  static ScratchPadManager manager;
  return &manager;
}
}  // namespace cpu_utils
