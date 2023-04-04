#pragma once

#include <cstddef>
#include <cassert>
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

#include "TypesAlias.hpp"

#include <cuda_runtime_api.h>

namespace GPU::Memory
{
    u32 static const DefaultAlign {8}; //64-bit aligned

    class LinearAllocator
    {
        private:
            uintptr_t const begin;
            uintptr_t current;
            uintptr_t const end;
        public:
            __host__ __device__ LinearAllocator(void * memory, u32 size);
            template<typename T>
            __host__ __device__ T * allocate(u32 size);
            __host__ __device__ void clear();
            __host__ __device__ void * getMemory() const;
            __host__ __device__ void * getFreeMemory() const;
            __host__ __device__ u32 getFreeMemorySize() const;
            __host__ __device__ u32 getUsedMemorySize() const;
            __host__ __device__ u32 getTotalMemorySize() const;
            __device__ static u32 getSharedMemorySize();
    };

    template<typename T>
    __host__ __device__ inline
    T * LinearAllocator::allocate(u32 size)
    {
        uintptr_t memory = current;
        u32 offset = memory % DefaultAlign;
        if (offset != 0)
        {
            memory += DefaultAlign - offset;
        }
        current = memory + size;
        assert(current < end);
        return reinterpret_cast<T *>(memory);
    }

    __host__ __device__ inline
    void LinearAllocator::clear()
    {
        current = begin;
    }

    __host__ __device__ inline
    void * LinearAllocator::getMemory() const
    {
        return reinterpret_cast<void *>(begin);
    }

    __host__ __device__ inline
    void * LinearAllocator::getFreeMemory() const
    {
        return reinterpret_cast<void *>(current);
    }

    __host__ __device__ inline
    u32 LinearAllocator::getFreeMemorySize() const
    {
        return static_cast<u32>(end - current);
    }

    __host__ __device__ inline
    u32 LinearAllocator::getUsedMemorySize() const
    {
        return static_cast<u32>(current - begin);
    }

    __host__ __device__ inline
    u32 LinearAllocator::getTotalMemorySize() const
    {
        return static_cast<u32>(end - begin);
    }

    __device__ inline
    u32 getSharedMemorySize()
    {
        u32 size;
        asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(size));
        return size;
    }

    template<typename T>
    T * mallocStd(u32 size)
    {
        void * memory = malloc(size);
        assert(memory != nullptr);
        return reinterpret_cast<T*>(memory);
    }

    template<typename T>
    T * mallocHost(u32 size)
    {
        void * memory = nullptr;
        cudaError_t status = cudaMallocHost(&memory, size);
        assert(status == cudaSuccess);
        assert(memory != nullptr);
        return reinterpret_cast<T*>(memory);
    }

    template<typename T>
    T * mallocDevice(u32 size)
    {
        void * memory = nullptr;
        cudaError_t status = cudaMalloc(&memory, size);
        assert(status == cudaSuccess);
        assert(memory != nullptr);
        return reinterpret_cast<T*>(memory);
    }

    template<typename T>
    T * mallocManaged(u32 size)
    {
        void * memory = nullptr;
        cudaError_t status = cudaMallocManaged(&memory, size);
        assert(status == cudaSuccess);
        assert(memory != nullptr);
        return reinterpret_cast<T*>(memory);
    }
}