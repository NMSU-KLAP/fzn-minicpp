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

#include <gpulib/Memory.cuh>

__host__ __device__
GPU::Memory::LinearAllocator::LinearAllocator(void * memory, u32 size) :
        begin(reinterpret_cast<uintptr_t>(memory)),
        current(begin),
        end(begin + static_cast<uintptr_t>(size))
{
    assert(begin < end);
}