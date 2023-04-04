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

#include <gpulib/Array.cuh>

namespace GPU
{
    template<typename T>
    class Vector : public Array<T>
    {
        // Members
        private:
            u32 _size;

        // Functions
        public:
            __host__ __device__ Vector(u32 capacity, T * data);
            __host__ __device__ T * at(u32 index) const;
            __host__ __device__ T * begin() const;
            __host__ __device__ T * end() const;
            __host__ __device__ u32 size() const;
            __host__ __device__ u32 capacity() const;
            __host__ __device__ void resize(u32 size);
            __host__ __device__ void clear();
            __host__ __device__ T * getData() const;
            __host__ __device__ u32 getDataSize() const;
            __host__ __device__ static u32 getDataSize(u32 capacity);
            __host__ __device__ Vector<T> & operator=(Vector<T> const & other);
            __host__ __device__ void push_back(T t);
            __host__ __device__ void print() const;
    };

    template<typename T>
    __host__ __device__
    Vector<T>::Vector(u32 capacity, T * data) :
            Array<T>(capacity, data),
            _size(0)
    {}

    template<typename T>
    __host__ __device__
    T * Vector<T>::at(u32 index) const
    {
        assert(index < _size);
        return Array<T>::at(index);
    }

    template<typename T>
    __host__ __device__
    T * Vector<T>::begin() const
    {
        return Array<T>::begin();
    }

    template<typename T>
    __host__ __device__
    T * Vector<T>::end() const
    {
        return Array<T>::begin() + _size;
    }

    template<typename T>
    __host__ __device__
    u32 Vector<T>::size() const
    {
        return _size;
    }

    template<typename T>
    __host__ __device__
    u32 Vector<T>::capacity() const
    {
        return Array<T>::capacity();
    }

    template<typename T>
    __host__ __device__
    Vector<T> & Vector<T>::operator=(Vector<T> const & other)
    {
        Array<T>::operator=(other);
        _size = other._size;
        return *this;
    }

    template<typename T>
    __host__ __device__ void
    Vector<T>::print() const
    {
        Array<T>::print(Array<T>::begin(), end());
    }

      template<typename T>
    __host__ __device__
    void Vector<T>::push_back(T t)
    {
        assert(_size < Array<T>::_capacity);
        _size += 1;
        *at(_size - 1) = t;
    }

    template<typename T>
    __host__ __device__
    void Vector<T>::clear()
    {
        resize(0);
    }

    template<typename T>
    __host__ __device__
    void Vector<T>::resize(u32 size)
    {
        assert(size == 0 or size <= Array<T>::_capacity);
        _size = size;
    }

    template<typename T>
    __host__ __device__
    T * Vector<T>::getData() const
    {
        return Array<T>::getData();
    }

    template<typename T>
    __host__ __device__
    u32 Vector<T>::getDataSize() const
    {
        return Array<T>::getDataSize(_size);
    }

    template<typename T>
    __host__ __device__
    u32 Vector<T>::getDataSize(u32 capacity)
    {
        return Array<T>::getDataSize(capacity);
    }
}
