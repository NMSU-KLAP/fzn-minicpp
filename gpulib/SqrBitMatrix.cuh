/*
 * fzn-minicpp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * fzn-minicpp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c) 2022. by Fabio Tardivo
 */

#pragma once

#include <gpulib/TypesAlias.hpp>
#include <gpulib/u128.cuh>

namespace GPU
{
    __host__ __device__ void cpyMatrixToBlock(u32 row, u32 col, u32 size, u32 const * matrix, u128 * block);

    __host__ __device__ void cpyArrangedToBlock(u32 row, u32 col, u32 size, u32 const * arranged, u128 * block);

    __host__ __device__ void cpyBlockToMatrix(u32 row, u32 col, u32 size, u128 const * block, u32 * matrix);

    __host__ __device__ void cpyBlockToArranged(u32 row, u32 col, u32 size, u128 const * block, u32 * arranged);

    __host__ __device__ void transposeBlock(u64 const * block, u64 * transposed);
    __host__ __device__ void transposeBlock(u128 const * block, u128 * transposed);

    __host__ __device__ void bitwiseAndBlock(u128 const * aBlock, u128 const * bBlock, u128 * cBlock);

    __host__ __device__ void findSccBlock(u32 row, u32 col, u128 const * block, u32 * scc);

    __host__ __device__ void reachabilityBlock(u128 * ijBlock, u128 const * iBlock, u128 const * kBlock);

    __host__ __device__ void reachabilityBlock3(u128 * ijBlock, u128 const * iBlock, u128 const * kBlock);

    __global__ void scc64(u32 const * graph, u32 * scc);
    __global__ void scc128(u32 const * adj, u32 * scc);

    void arrange(u32 size, u32 const * matrix, u32 * arranged);
    __global__ void arrangeKernel(u32 size, u32 const * matrix, u32 * arranged);

    void dearrange(u32 size, u32 const * arranged, u32 * matrix);
    __global__ void dearrangeKernel(u32 size, u32 const * arranged, u32 * matrix);

    void transposeArranged(u32 size, u32 const * arranged, u32 * transposed);
    __global__ void transposeArrangedKernel(u32 size, u32 const * arranged, u32 * transposed);

    void bitwiseAnd(u32 size, u32 const * a, u32 const * b, u32 * c);
    __global__ void bitwiseAndKernel(u32 size, u32 const * a, u32 const * b, u32 * c);

    void reachabilityArranged(u32 size, u32 * reach);
    __global__ void reachabilityArrangedKernel1(u32 i, u32 size, u32 * reach);
    __global__ void reachabilityArrangedKernel2(u32 i, u32 size, u32 * reach);
    __global__ void reachabilityArrangedKernel3(u32 i, u32 size, u32 * reach);

    void initScc(u32 size, u32 * scc);
    __global__ void initSccKernel(u32 size, u32 * scc);
    void findSccArranged(u32 size, u32 const * a, u32 * scc);
    __global__ void findSccArrangedKernel(u32 size, u32 const * a, u32 * scc);

    void debugWorkflow();
    void debugReachability(u32 size, u32 * matrix);
    void profileReachability();
}