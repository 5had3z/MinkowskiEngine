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

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace minkowski
{

    template <typename coordinate_type>
    std::pair<at::Tensor, at::Tensor> LocalPoolingTransposeForwardCPU(
        at::Tensor const &in_feat,
        default_types::stride_type const &kernel_size,     //
        default_types::stride_type const &kernel_stride,   //
        default_types::stride_type const &kernel_dilation, //
        RegionType::Type const region_type,                //
        at::Tensor const &offset,                          //
        bool generate_new_coordinates,                     //
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
            auto map_it = p_map_manager->find(p_in_map_key->get_key());
            TORCH_CHECK(map_it != p_map_manager->map_end(), ERROR_MAP_NOT_FOUND);
            auto const &in_map = (*map_it).second;

            auto out_tensor_stride = detail::stride_tensor_stride(
                in_map.get_tensor_stride(), kernel_stride, true /* is_transpose */);
            auto kernel_region = cpu_kernel_region<coordinate_type>(
                region_type,              //
                in_map.coordinate_size(), //
                out_tensor_stride.data(), //
                kernel_size.data(),       //
                kernel_dilation.data(),   //
                0,                        // volume. Will be initialized automatically
                offset.data_ptr<coordinate_type>(), offset.size(0),
                true // is_transpose
            );

            coordinate_map_key_type out_key = std::get<0>(p_map_manager->stride_region(
                in_key, kernel_region, out_tensor_stride, generate_new_coordinates));
            p_out_map_key->set_key(out_key);
        }

        cpu_kernel_map const &in_out = p_map_manager->kernel_map(
            p_in_map_key,    //
            p_out_map_key,   //
            kernel_size,     //
            kernel_stride,   //
            kernel_dilation, //
            region_type,     //
            offset, true /* is_transpose */, true /* is_pool */);

        auto const out_nrows = p_map_manager->size(p_out_map_key->get_key());
        at::Tensor out_feat =
            torch::zeros({out_nrows, in_feat.size(1)}, in_feat.options());
        LOG_DEBUG("Allocated", out_nrows, "x", in_feat.size(1), "features.");

        at::Tensor num_nonzero =
            torch::empty({0}, in_feat.options().requires_grad(false));
        AT_DISPATCH_FLOATING_TYPES(
            in_feat.scalar_type(), "local_pooling_forward_cpu", [&]
            { NonzeroAvgPoolingForwardKernelCPU<scalar_t, coordinate_type>(
                  in_feat.template data_ptr<scalar_t>(),
                  out_feat.template data_ptr<scalar_t>(),
                  num_nonzero.template data_ptr<scalar_t>(), in_feat.size(1),
                  in_out.first, in_out.second, out_nrows, false); });
        return std::make_pair(out_feat, num_nonzero);
    }

    template <typename coordinate_type>
    at::Tensor LocalPoolingTransposeBackwardCPU(
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
            offset, true /* is_transpose */, true /* is_pool */);

        at::Tensor grad_in_feat =
            torch::zeros({in_feat.size(0), in_feat.size(1)}, in_feat.options());

        AT_DISPATCH_FLOATING_TYPES(
            in_feat.scalar_type(), "local_pooling_backward_cpu", [&]
            { NonzeroAvgPoolingBackwardKernelCPU<scalar_t, default_types::index_type>(
                  grad_in_feat.template data_ptr<scalar_t>(), in_feat.size(0),
                  grad_out_feat.template data_ptr<scalar_t>(),
                  num_nonzero.template data_ptr<scalar_t>(), in_feat.size(1),
                  in_out.first, in_out.second, false /* avg */); });
        return grad_in_feat;
    }

    template std::pair<at::Tensor, at::Tensor>
    LocalPoolingTransposeForwardCPU<int32_t>(
        at::Tensor const &in_feat,
        default_types::stride_type const &kernel_size,     //
        default_types::stride_type const &kernel_stride,   //
        default_types::stride_type const &kernel_dilation, //
        RegionType::Type const region_type,                //
        at::Tensor const &offset,                          //
        bool generate_new_coordinates,                     //
        PoolingMode::Type pooling_mode,                    //
        CoordinateMapKey *p_in_map_key,                    //
        CoordinateMapKey *p_out_map_key,                   //
        cpu_manager_type<int32_t> *p_map_manager);

    template at::Tensor LocalPoolingTransposeBackwardCPU<int32_t>(
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
