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
#include <gpulib/TypesAlias.hpp>
#include <gpulib/Memory.cuh>
#include <cstddef>
#include <cstdio>

namespace GPU
{
    template<typename T>
    class Array
    {
        // Members
        protected:
            u32 _capacity;
            T * _data;

        // Functions
        public:
            __host__ __device__ Array();
            __host__ __device__ Array(u32 capacity, T * data);
            __host__ __device__ T * at(u32 index) const;
            __host__ __device__ T * begin() const;
            __host__ __device__ T * end() const;
            __host__ __device__ T * first() const;
            __host__ __device__ T * last() const;
            __host__ __device__ u32 capacity() const;
            __host__ __device__ T * getData() const;
            __host__ __device__ u32 getDataSize() const;
            __host__ __device__ static u32 getDataSize(u32 capacity);
            __host__ __device__ Array<T> & operator=(Array<T> const & other);
            __host__ __device__ void print() const;
            __host__ __device__ static void print(T const * begin, T const * end);
    };

    template<typename T>
    __host__ __device__ inline
    Array<T>::Array() :
            _capacity(),
            _data(nullptr)
    {}

    template<typename T>
    __host__ __device__
    Array<T>::Array(u32 capacity, T * data) :
            _capacity(capacity),
            _data(data)
    {}

    template<typename T>
    __host__ __device__ inline
    T * Array<T>::at(u32 index) const
    {
        assert(_capacity > 0);
        assert(index < _capacity);
        return _data + index;
    }

    template<typename T>
    __host__ __device__ inline
    T * Array<T>::begin() const
    {
        return _data;
    }

    template<typename T>
    __host__ __device__ inline
    T * Array<T>::first() const
    {
        return _data;
    }

    template<typename T>
    __host__ __device__ inline
    T * Array<T>::last() const
    {
        return _data + (_capacity - 1);
    }

    template<typename T>
    __host__ __device__ inline
    T * Array<T>::getData() const
    {
        return _data;
    }

    template<typename T>
    __host__ __device__ inline
    T * Array<T>::end() const
    {
        return begin() + _capacity;
    }

    template<typename T>
    __host__ __device__ inline
    u32 Array<T>::capacity() const
    {
        return _capacity;
    }

    template<typename T>
    __host__ __device__ inline
    Array<T> & Array<T>::operator=(Array<T> const & other)
    {
        _capacity = other._capacity;
        _data = other._data;
        return *this;
    }

    template<typename T>
    __host__ __device__
    void Array<T>::print() const
    {
        print(begin(), end());
    }

    template<typename T>
    __host__ __device__
    void Array<T>::print(T const * begin, T const * end)
    {
        static_assert(std::is_integral<T>::value);
        for(T const * t = begin; t != end; t += 1)
        {
            printf("%d ", *t);
        }
        printf("\n");
    }

    template<typename T>
    __host__ __device__
    u32 Array<T>::getDataSize() const
    {
       return getDataSize(_capacity);
    }

    template<typename T>
    __host__ __device__
    u32 Array<T>::getDataSize(u32 capacity)
    {
        return sizeof(T) * capacity;
    }
}
