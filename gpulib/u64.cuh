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

#include "TypesAlias.hpp"

__host__ __device__
inline bool getBit(u64 const & n, u32 b)
{
    assert(b < 64);
    u32 const * tmp = reinterpret_cast<u32 const *>(&n);
    u32 const wordIdx = 1 - (b >> 5);
    u32 const bitIdx = b % 32;
    u32 const mask = 1 << bitIdx;
    return (tmp[wordIdx] & mask) != 0;
}

__host__ __device__
inline void orBit(u64 & n, u32 b, bool value)
{
    assert(b < 64);

    u32 * const tmp = reinterpret_cast<u32 *>(&n);
    u32 const wordIdx = 1 - (b >> 5);
    u32 const bitIdx = b & 31;
    u32 const mask = static_cast<u32>(value) << bitIdx;
    tmp[wordIdx] |= mask;
}