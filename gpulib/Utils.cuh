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

#include <cassert>
#include "TypesAlias.hpp"
#include <gpulib/Vector.cuh>

namespace GPU
{
    template<typename T>
    __host__ __device__ T min(T const & a, T const & b);

    template<typename T>
    __host__ __device__ T max(T const & a, T const & b);

    template<typename T>
    __host__ __device__ T ceilDivision(T const & a, T const & b);

    template<typename T>
    __host__ __device__ T roundDown(T const & a, T const & b);

    __host__ __device__ void getBeginEnd(u32 * begin, u32 *end, u32 index, u32 workers, u32 jobs);

    template<typename T>
    __host__ __device__ u32 partition(T * values, u32 firstIndex, u32 lastIndex);

    template<typename T>
    __host__ __device__ void quickSort(T * values, u32 firstIndex, u32 lastIndex);

    template<typename T>
    __host__ __device__ void swap(T & a, T & b);

    template<typename T>
    __device__ bool sortedInsertionNotDuplicates(GPU::Vector<T> & vector, T t);

    __device__ unsigned int inline getLeftmostOneIndex64(unsigned long long int const & val);
}

template<typename T>
__host__ __device__ inline
T GPU::min(T const & a, T const & b)
{
    static_assert(std::is_arithmetic_v<T>);
    return a <= b ? a : b;
}

template<typename T>
__host__ __device__ inline
T GPU::max(T const & a, T const & b)
{
    static_assert(std::is_arithmetic_v<T>);
    return a >= b ? a : b;
}

__host__ __device__ inline
void GPU::getBeginEnd(u32 * begin, u32 *end, u32 index, u32 workers, u32 jobs)
{
    u32 const jobsPerWorker = ceilDivision(jobs, workers);
    *begin = jobsPerWorker * index;
    *end = GPU::min(jobs, *begin + jobsPerWorker);
}


template<typename T>
__host__ __device__ inline
T GPU::ceilDivision(T const & a, T const & b)
{
    static_assert(std::is_unsigned_v<T>);
    return (a + b - 1) / b;
}

template<typename T>
__host__ __device__ inline
T GPU::roundDown(T const & a, T const & b)
{
    static_assert(std::is_integral_v<T>);
    return (a / b) * b;
}

template<typename T>
__host__ __device__
u32 GPU::partition(T * values, u32 firstIndex, u32 lastIndex)
{
    static_assert(std::is_arithmetic_v<T>);

    i32 pivotValue = values[(firstIndex + lastIndex) / 2];
    i32 bottomIndex = firstIndex - 1;
    i32 topIndex = lastIndex + 1;

    while (true)
    {
        do
        {
            bottomIndex += 1;
        }
        while(values[bottomIndex] < pivotValue);

        do
        {
            topIndex -= 1;
        }
        while(values[topIndex] > pivotValue);

        if (bottomIndex >= topIndex)
        {
            return topIndex;
        }

        GPU::swap(values[bottomIndex], values[topIndex]);
    }
}

template<typename T>
__host__ __device__
void GPU::quickSort(T * values, u32 firstIndex, u32 lastIndex)
{
    static_assert(std::is_arithmetic_v<T>);

    if (firstIndex < lastIndex)
    {
        i32 pivotIndex = partition(values, firstIndex, lastIndex);
        quickSort(values, firstIndex, pivotIndex);
        quickSort(values, pivotIndex + 1, lastIndex);
    }
}

template<typename T>
__host__ __device__ inline
void GPU::swap(T & a, T & b)
{
    static_assert(std::is_scalar_v<T>);
    T tmp = a;
    a = b;
    b = tmp;
}

template<typename T>
__device__
bool GPU::sortedInsertionNotDuplicates(GPU::Vector<T> & vector, T t)
{
    static_assert(std::is_arithmetic_v<T>);
    assert(vector.size() < vector.capacity());

    vector.push_back(t);
    for(i32 i = vector.size() - 1; i > 0; i -= 1)
    {
        T & t0 = *vector.at(i-1);
        T & t1 = *vector.at(i);
        if(t0 > t1)
        {
            swap(t0,t1);
        }
        else if (t0 == t1)
        {
            return false;
        }
        else
        {
            break;
        }
    }
    return true;
}

__device__
unsigned int GPU::getLeftmostOneIndex64(unsigned long long int const & val)
{
    return __clzll(val);
}
