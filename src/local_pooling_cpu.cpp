/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */
#include "coordinate_map.hpp"
#include "coordinate_map_cpu.hpp"
#include "coordinate_map_key.hpp"
#include "coordinate_map_manager.hpp"
#include "errors.hpp"
#include "types.hpp"
#include "utils.hpp"

#include "pooling_avg_kernel.hpp"
#include "pooling_max_kernel.hpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace minkowski
{

    template <typename coordinate_type>
    std::pair<at::Tensor, at::Tensor>
    LocalPoolingForwardCPU(at::Tensor const &in_feat,
                           default_types::stride_type const &kernel_size,     //
                           default_types::stride_type const &kernel_stride,   //
                           default_types::stride_type const &kernel_dilation, //
                           RegionType::Type const region_type,                //
                           at::Tensor const &offset,                          //
                           PoolingMode::Type pooling_mode,                    //
                           CoordinateMapKey *p_in_map_key,                    //
                           CoordinateMapKey *p_out_map_key,                   //
                           cpu_manager_type<coordinate_type> *p_map_manager)
    {

        TORCH_CHECK(in_feat.is_contiguous(), "in_feat must be contiguous");
        TORCH_CHECK(!in_feat.is_cuda(), "in_feat must be CPU");
        TORCH_CHECK(in_feat.dim() == 2, "in_feat.dim():", in_feat.dim());

        coordinate_map_key_type in_key = p_in_map_key->get_key();
        TORCH_CHECK(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);

        TORCH_CHECK(in_feat.size(0) == p_map_manager->size(in_key), "Invalid in_feat size",
                    in_feat.size(0), "!=", p_map_manager->size(in_key));

        // create an output coordinate map
        if (!p_out_map_key->is_key_set())
        {
            coordinate_map_key_type out_key =
                std::get<0>(p_map_manager->stride(in_key, kernel_stride));
            p_out_map_key->set_key(out_key);
        }

        cpu_kernel_map const &in_out = p_map_manager->kernel_map(
            p_in_map_key,    //
            p_out_map_key,   //
            kernel_size,     //
            kernel_stride,   //
            kernel_dilation, //
            region_type,     //
            offset, false /* is_transpose */, true /* is_pool */);

        auto const out_nrows = p_map_manager->size(p_out_map_key->get_key());
        at::Tensor out_feat =
            torch::zeros({out_nrows, in_feat.size(1)}, in_feat.options());
        LOG_DEBUG("Allocated", out_nrows, "x", in_feat.size(1), "features.");

        if (pooling_mode == PoolingMode::LOCAL_MAX_POOLING)
        {
            at::Tensor max_index = torch::empty(
                {0}, in_feat.options().dtype(torch::kInt).requires_grad(false));
            max_index.resize_({out_nrows, in_feat.size(1)});
            max_index.zero_();

            AT_DISPATCH_FLOATING_TYPES(
                in_feat.scalar_type(), "local_pooling_forward_cpu", [&]
                { MaxPoolingForwardKernelCPU<scalar_t, int32_t,
                                             default_types::index_type>(
                      in_feat.template data_ptr<scalar_t>(),
                      out_feat.template data_ptr<scalar_t>(),
                      max_index.data_ptr<int32_t>(), in_feat.size(1), in_out.first,
                      in_out.second, out_nrows); });
            return std::make_pair(out_feat, max_index);
        }
        else
        {
            at::Tensor num_nonzero =
                torch::empty({0}, in_feat.options().requires_grad(false));
            if (pooling_mode == PoolingMode::LOCAL_AVG_POOLING)
            {
                num_nonzero.resize_({out_nrows});
                num_nonzero.zero_();
            }
            AT_DISPATCH_FLOATING_TYPES(
                in_feat.scalar_type(), "local_pooling_forward_cpu", [&]
                { NonzeroAvgPoolingForwardKernelCPU<scalar_t, coordinate_type>(
                      in_feat.template data_ptr<scalar_t>(),
                      out_feat.template data_ptr<scalar_t>(),
                      num_nonzero.template data_ptr<scalar_t>(), in_feat.size(1),
                      in_out.first, in_out.second, out_nrows, pooling_mode); });
            return std::make_pair(out_feat, num_nonzero);
        }
    }

    template <typename coordinate_type>
    at::Tensor
    LocalPoolingBackwardCPU(at::Tensor const &in_feat,                         //
                            at::Tensor const &grad_out_feat,                   //
                            at::Tensor const &num_nonzero,                     //
                            default_types::stride_type const &kernel_size,     //
                            default_types::stride_type const &kernel_stride,   //
                            default_types::stride_type const &kernel_dilation, //
                            RegionType::Type const region_type,                //
                            at::Tensor const &offset,                          //
                            PoolingMode::Type pooling_mode,                    //
                            CoordinateMapKey *p_in_map_key,                    //
                            CoordinateMapKey *p_out_map_key,                   //
                            cpu_manager_type<coordinate_type> *p_map_manager)
    {
        TORCH_CHECK(in_feat.is_contiguous(), "in_feat must be contiguous");
        TORCH_CHECK(grad_out_feat.is_contiguous(), "grad_out_feata must be contiguous");

        TORCH_CHECK(!in_feat.is_cuda(), "in_feat must be CPU");
        TORCH_CHECK(!grad_out_feat.is_cuda(), "in_feat must be CPU");

        TORCH_CHECK(in_feat.scalar_type() == grad_out_feat.scalar_type(), "type mismatch");

        TORCH_CHECK(in_feat.dim() == 2, "in_feat.dim():", in_feat.dim());
        TORCH_CHECK(grad_out_feat.dim() == 2, "grad_out_feat.dim():", grad_out_feat.dim());

        coordinate_map_key_type in_key = p_in_map_key->get_key();
        TORCH_CHECK(p_map_manager->exists(in_key), ERROR_MAP_NOT_FOUND);
        coordinate_map_key_type out_key = p_out_map_key->get_key();
        TORCH_CHECK(p_map_manager->exists(out_key), ERROR_MAP_NOT_FOUND);

        cpu_kernel_map const &in_out = p_map_manager->kernel_map(
            p_in_map_key,    //
            p_out_map_key,   //
            kernel_size,     //
            kernel_stride,   //
            kernel_dilation, //
            region_type,     //
            offset, false /* is_transpose */, true /* is_pool */);

        at::Tensor grad_in_feat =
            torch::zeros({in_feat.size(0), in_feat.size(1)}, in_feat.options());

        if (pooling_mode == PoolingMode::LOCAL_MAX_POOLING)
        {
            AT_DISPATCH_FLOATING_TYPES(
                in_feat.scalar_type(), "local_pooling_backward_cpu", [&]
                { MaxPoolingBackwardKernelCPU<scalar_t, int32_t>(
                      grad_in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
                      grad_out_feat.template data_ptr<scalar_t>(),
                      grad_out_feat.size(0), num_nonzero.data_ptr<int32_t>(),
                      in_feat.size(1)); });
        }
        else
        {
            AT_DISPATCH_FLOATING_TYPES(
                in_feat.scalar_type(), "local_pooling_backward_cpu", [&]
                { NonzeroAvgPoolingBackwardKernelCPU<scalar_t,
                                                     default_types::index_type>(
                      grad_in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
                      grad_out_feat.template data_ptr<scalar_t>(),
                      num_nonzero.template data_ptr<scalar_t>(), in_feat.size(1),
                      in_out.first, in_out.second, pooling_mode); });
        }
        return grad_in_feat;
    }

    // int32
    template void max_pooling_forward_pointer_kernel_cpu<float, int32_t, int32_t>(
        float const *p_in_feat, float *p_out_feat, int32_t *p_mask_index,
        size_t const nchannel,
        int32_t const *const p_in_maps,  //
        int32_t const *const p_out_maps, //
        size_t const map_size);

    template void max_pooling_forward_pointer_kernel_cpu<double, int32_t, int32_t>(
        double const *p_in_feat, double *p_out_feat, int32_t *p_mask_index,
        size_t const nchannel,
        int32_t const *const p_in_maps,  //
        int32_t const *const p_out_maps, //
        size_t const map_size);

    template void MaxPoolingBackwardKernelCPU<float, int32_t>(
        float *p_grad_in_feat, size_t const in_nrows, float const *p_grad_out_feat,
        size_t const out_nrows, int32_t const *p_mask_index, size_t const nchannel);

    template void MaxPoolingBackwardKernelCPU<double, int32_t>(
        double *p_grad_in_feat, size_t const in_nrows,
        double const *p_grad_out_feat, size_t const out_nrows,
        int32_t const *p_mask_index, size_t const nchannel);

    // int64
    template void max_pooling_forward_pointer_kernel_cpu<float, int64_t, int64_t>(
        float const *p_in_feat, float *p_out_feat, int64_t *p_mask_index,
        size_t const nchannel,
        int64_t const *const p_in_maps,  //
        int64_t const *const p_out_maps, //
        size_t const map_size);

    template void max_pooling_forward_pointer_kernel_cpu<double, int64_t, int64_t>(
        double const *p_in_feat, double *p_out_feat, int64_t *p_mask_index,
        size_t const nchannel,
        int64_t const *const p_in_maps,  //
        int64_t const *const p_out_maps, //
        size_t const map_size);

    template void MaxPoolingBackwardKernelCPU<float, int64_t>(
        float *p_grad_in_feat, size_t const in_nrows, float const *p_grad_out_feat,
        size_t const out_nrows, int64_t const *p_mask_index, size_t const nchannel);

    template void MaxPoolingBackwardKernelCPU<double, int64_t>(
        double *p_grad_in_feat, size_t const in_nrows,
        double const *p_grad_out_feat, size_t const out_nrows,
        int64_t const *p_mask_index, size_t const nchannel);

    template std::pair<at::Tensor, at::Tensor> LocalPoolingForwardCPU<int32_t>(
        at::Tensor const &in_feat,
        default_types::stride_type const &kernel_size,     //
        default_types::stride_type const &kernel_stride,   //
        default_types::stride_type const &kernel_dilation, //
        RegionType::Type const region_type,                //
        at::Tensor const &offset,                          //
        PoolingMode::Type pooling_mode,                    //
        CoordinateMapKey *p_in_map_key,                    //
        CoordinateMapKey *p_out_map_key,                   //
        cpu_manager_type<int32_t> *p_map_manager);

    template at::Tensor LocalPoolingBackwardCPU<int32_t>(
        at::Tensor const &in_feat,                         //
        at::Tensor const &grad_out_feat,                   //
        at::Tensor const &num_nonzero,                     //
        default_types::stride_type const &kernel_size,     //
        default_types::stride_type const &kernel_stride,   //
        default_types::stride_type const &kernel_dilation, //
        RegionType::Type const region_type,                //
        at::Tensor const &offset,                          //
        PoolingMode::Type pooling_mode,                    //
        CoordinateMapKey *p_in_map_key,                    //
        CoordinateMapKey *p_out_map_key,                   //
        cpu_manager_type<int32_t> *p_map_manager);

} // end namespace minkowski
