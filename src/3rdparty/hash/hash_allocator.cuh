/*
 * Copyright (c) 2017, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HASH_ALLOCATOR_CUH
#define HASH_ALLOCATOR_CUH

#include <new>
#include <thrust/device_reference.h>

template <class T>
struct managed_allocator
{
  typedef T value_type;
  // rmm::mr::device_memory_resource* mr = new rmm::mr::managed_memory_resource;

  managed_allocator() = default;

  template <class U>
  constexpr managed_allocator(const managed_allocator<U> &) noexcept
  {
  }

  T *allocate(std::size_t n, cudaStream_t stream = 0) const
  {
    T *d_tmp;
    cudaError_t error = cudaMalloc((void **)&d_tmp, n * sizeof(T));
    if (error != cudaSuccess)
    {
      cudaGetLastError(); // clear CUDA error
      std::runtime_error("cudaMalloc failed in the hash_allocator.cuh:managed_allocator.");
    }
    return d_tmp;
    // return static_cast<T*>(mr->allocate(n * sizeof(T), stream));
  }

  void deallocate(T *p, std::size_t n, cudaStream_t stream = 0) const
  {
    cudaFree(p);
    // mr->deallocate(p, n * sizeof(T), stream);
  }
};

template <class T, class U>
bool operator==(const managed_allocator<T> &, const managed_allocator<U> &)
{
  return true;
}
template <class T, class U>
bool operator!=(const managed_allocator<T> &, const managed_allocator<U> &)
{
  return false;
}

template <class T>
struct default_allocator
{
  typedef T value_type;
  // rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource();

  default_allocator() = default;

  template <class U>
  constexpr default_allocator(const default_allocator<U> &) noexcept
  {
  }

  T *allocate(std::size_t n, cudaStream_t stream = 0) const
  {
    T *d_tmp;
    cudaError_t error = cudaMalloc((void **)&d_tmp, n * sizeof(T));
    if (error != cudaSuccess)
    {
      cudaGetLastError(); // clear CUDA error
      std::runtime_error("cudaMalloc failed in the hash_allocator.cuh:default_allocator.");
    }

    return d_tmp;
    // return static_cast<T*>(mr->allocate(n * sizeof(T), stream));
  }

  void deallocate(T *p, std::size_t n, cudaStream_t stream = 0) const
  {
    cudaFree(p);
    // mr->deallocate(p, n * sizeof(T), stream);
  }
};

template <class T, class U>
bool operator==(const default_allocator<T> &, const default_allocator<U> &)
{
  return true;
}
template <class T, class U>
bool operator!=(const default_allocator<T> &, const default_allocator<U> &)
{
  return false;
}

#endif
