/*
Copyright (c) Meta Platforms, Inc. and affiliates.

All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

#pragma once

#include <array>
#include <vector>

#include "gpu.cuh"
#include "kernel_map.cuh"
#include "math_functions.cuh"
#include "types.hpp"

namespace minkowski
{

    template <typename Dtype, typename Itype, typename ByteAllocator>
    void DepthwiseConvolution2ForwardKernelGPU(
        Dtype const *d_in_feat,                      //
        default_types::size_type const in_nchannel,  //
        Dtype *d_out_feat,                           //
        default_types::size_type const out_nchannel, //
        Dtype *d_kernel, gpu_kernel_map<Itype, ByteAllocator> const &kernel_map,
        cudaStream_t stream);

    template <typename Dtype, typename Itype, typename ByteAllocator>
    void DepthwiseConvolution2BackwardKernelGPU(
        Dtype const *d_in_feat,                      //
        Dtype *d_grad_in_feat,                       //
        default_types::size_type const in_nchannel,  //
        Dtype const *d_grad_out_feat,                //
        default_types::size_type const out_nchannel, //
        Dtype const *d_kernel,                       //
        Dtype *d_grad_kernel,                        //
        gpu_kernel_map<Itype, ByteAllocator> const &kernel_map,
        cudaStream_t stream);
} // end namespace minkowski
