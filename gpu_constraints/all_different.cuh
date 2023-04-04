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

#include <vector>
#include <intvar.hpp>
#include <constraint.hpp>
#include <gpulib/Array.cuh>
#include <gpulib/BitMatrix.cuh>

class AllDifferentGPU : public Constraint
{
    private:
        std::vector<var<int>::Ptr> const vars;
        u32 _nVar;
        u32 _nVal;
        i32 _minVal;
        i32 _maxVal;
        u32 _nNodes;
        AllDifferentAC allDifferentAc;
        GPU::BitMatrix * graph_h;
        u32* matrix1_d;
        u32* matrix2_d;
        u32* matrix3_d;
        GPU::Array<u32> * scc_h;
        u32* scc_d;
        cudaStream_t cuStream;
        u32 nEdges;
        cudaGraph_t graph;
        cudaGraphExec_t graph2sccLowLatency;
        u32 iteration;

    public:
        AllDifferentGPU(std::vector<var<int>::Ptr> const & vars);
        void updateBounds();
        void post() override;
        void propagate() override;
    private:
        void initGraph2sccLowLatency();
        void domains2graph(int * match);
        void graph2scc();
};