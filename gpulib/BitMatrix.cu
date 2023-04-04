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

# include <gpulib/BitMatrix.cuh>

GPU::BitMatrix::BitMatrix(u32 rows, u32 cols, u32 * data) :
        Array<u32>(rows * (cols / 32), data),
        _rows(rows),
        _cols(cols)
{
    assert(_rows > 0);
    assert(_cols > 0);
    assert(_rows == 64 or _rows % 128 == 0);
    assert(_cols == 64 or _cols % 128 == 0);
}

__host__ __device__
void GPU::BitMatrix::print(u32 rows, u32 cols, u32 const * data)
{
    for (u32 row = 0; row < rows; row += 1)
    {
        for (u32 col = 0; col < cols; col += 1)
        {
            printf("%c ", get(row, col, rows, cols, data) ? '1' : '0');
        }
        printf("\n");
    }
#ifndef __CUDA_ARCH__
    fflush(stdout);
#endif
}

void GPU::BitMatrix::clear()
{
    std::fill(Array<u32>::begin(), Array<u32>::end(), 0);
}
