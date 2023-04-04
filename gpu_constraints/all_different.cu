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

#include <new>
#include <algorithm>
#include <chrono>
#include <gpu_constraints/all_different.cuh>
#include <gpulib/SqrBitMatrix.cuh>
#include <gpulib/Memory.cuh>

using namespace GPU::Memory;
using namespace std::chrono;

AllDifferentGPU::AllDifferentGPU(std::vector<var<int>::Ptr> const & vars) :
        Constraint(vars[0]->getSolver()),
        vars(vars),
        allDifferentAc(vars)
{
    setPriority(CLOW);
    _nVar = vars.size();
    updateBounds();
    iteration = 0;

    //Initialize memory
    u32 * mem = mallocHost<u32>(GPU::BitMatrix::getDataSize(_nNodes, _nNodes));
    graph_h = new GPU::BitMatrix(_nNodes, _nNodes, mem);

    matrix1_d = mallocDevice<u32>(graph_h->getDataSize());
    matrix2_d = mallocDevice<u32>(graph_h->getDataSize());
    matrix3_d = mallocDevice<u32>(graph_h->getDataSize());

    mem = mallocHost<u32>(GPU::Array<u32>::getDataSize(_nNodes));
    scc_h = new GPU::Array<u32>(_nNodes, mem);
    scc_d = mallocDevice<u32>(GPU::Array<u32>::getDataSize(_nNodes));

    cudaStreamCreate(&cuStream);
    initGraph2sccLowLatency();
}

void AllDifferentGPU::updateBounds()
{
    _minVal = INT32_MAX;
    _maxVal = INT32_MIN;
    for(u32 i = 0; i < _nVar; i += 1)
    {
        _minVal = std::min(_minVal, vars[i]->min());
        _maxVal = std::max(_maxVal, vars[i]->max());
    }
    _minVal = floorDivision(_minVal, 32) * 32;
    _maxVal = _minVal + ceilDivision(_maxVal - _minVal + 1, 32) * 32 - 1;
    _nVal = _maxVal - _minVal + 1;
    _nNodes = _nVal + _nVar + 1;
    if (_nNodes <= 64)
    {
        _nNodes = 64;
    }
    else if (_nNodes <= 128)
    {
        _nNodes = 128;
    }
    else
    {
        _nNodes = ceilDivision(_nNodes, 128) * 128;
    }
}

void AllDifferentGPU::post()
{
    allDifferentAc.init();
    for (u32 i = 0; i < _nVar; i += 1)
    {
        vars[i]->propagateOnDomainChange(this);
    }
    propagate();
}
void AllDifferentGPU::initGraph2sccLowLatency()
{
    cudaStreamBeginCapture(cuStream, cudaStreamCaptureModeGlobal);
    graph2scc();
    cudaStreamEndCapture(cuStream, &graph);
    cudaGraphInstantiate(&graph2sccLowLatency, graph, nullptr, nullptr, 0);
}

void AllDifferentGPU::propagate()
{
    //Timer::begin("AllDifferentGPU");
    if ((u32) allDifferentAc.calcMatch() < _nVar)
    {
        //Timer::end("AllDifferentGPU");
        failNow();
    }

    iteration += 1;
    nEdges = 0;
    for (u32 i  = 0; i < _nVar; i += 1)
    {
        nEdges += vars[i]->size();
    }

    if (false) //nEdges <= 256 * 256)
    {
        auto start = high_resolution_clock::now();
        allDifferentAc.calcSCC();
        auto stop = high_resolution_clock::now();
        auto durationCPU = duration_cast<microseconds>(stop - start);

        start = high_resolution_clock::now();
        domains2graph(allDifferentAc.getMatch());
        graph2scc();
        cudaGraphLaunch(graph2sccLowLatency, cuStream);
        cudaStreamSynchronize(cuStream);
        stop = high_resolution_clock::now();
        auto durationGPU = duration_cast<microseconds>(stop - start);

        double ratio = (double) nEdges / (double) (_nNodes * _nNodes);
        if (iteration % 100 == 0)
        {
            iteration = 0;
            printf("%%%% , %d, %.5f, %ld, %ld\n", nEdges, ratio, durationCPU.count(), durationGPU.count());
            fflush(stdout);
        }

        allDifferentAc.filterDomains();

    }
    else
    {
       // Timer::begin("SCCsGPU");
        int * match = allDifferentAc.getMatch();
        domains2graph(match);
        //graph2scc();
        cudaGraphLaunch(graph2sccLowLatency, cuStream);
        cudaStreamSynchronize(cuStream);
        //Timer::end("SCCsGPU");

        for (u32 var = 0; var < _nVar; var += 1)
        {
            i32 const minVal = vars[var]->min();
            i32 const maxVal = vars[var]->max();
            for (i32 val = minVal; val <= maxVal; val += 1)
            {
                u32 const varNode = _nVal + var;
                u32 const valNode = val - _minVal;
                if (match[var] != val and *scc_h->at(varNode) != *scc_h->at(valNode))
                {
                    vars[var]->remove(val);
                }
            }
        }
    }
    //Timer::end("AllDifferentGPU");
}

void AllDifferentGPU::domains2graph(int * match)
{
    graph_h->clear();

    // Edges variables -> values
    for(u32 var = 0u; var < _nVar; var += 1)
    {
        u32 const varNode = _nVal + var;
        vars[var]->dump(_minVal, _maxVal, graph_h->getRow(varNode));
    }

    // Edges sink -> values
    u32 const sinkNode = _nVal + _nVar;
    for (i32 val = _minVal; val <= _maxVal; val += 1)
    {
        u32 const valNode = val - _minVal;
        graph_h->set(valNode, sinkNode, true);
    }

    // Match edges
    for (u32 var = 0; var < _nVar; var += 1)
    {
        u32 const valNode = match[var] - _minVal;
        u32 const varNode = _nVal + var;

        // Edges variables <-> values
        graph_h->set(varNode, valNode, false);
        graph_h->set(valNode, varNode, true);

        // Edges sink <-> values
        graph_h->set(valNode, sinkNode, false);
        graph_h->set(sinkNode, valNode, true);
    }
}

void AllDifferentGPU::graph2scc()
{
    u32 * reach = matrix1_d;
    u32 * reach_ = matrix2_d;
    u32 * reach_t = matrix3_d;

    if (_nNodes == 64)
    {
        cudaMemcpyAsync(reach, graph_h->getData(), graph_h->getDataSize(), cudaMemcpyHostToDevice, cuStream);
        GPU::scc64<<<1,64,0, cuStream>>>(reach, scc_d);
        cudaMemcpyAsync(scc_h->getData(), scc_d, scc_h->getDataSize(), cudaMemcpyDeviceToHost, cuStream);
        cudaStreamSynchronize(cuStream);
    }
    else if (_nNodes == 128)
    {
        cudaMemcpyAsync(reach, graph_h->getData(), graph_h->getDataSize(), cudaMemcpyHostToDevice, cuStream);
        GPU::scc128<<<1,128,0, cuStream>>>(reach, scc_d);
        cudaMemcpyAsync(scc_h->getData(), scc_d, scc_h->getDataSize(), cudaMemcpyDeviceToHost, cuStream);
        cudaStreamSynchronize(cuStream);
    }
    else
    {
        u32 const blocks = _nNodes / 128;
        dim3 const dimBlock2 = dim3(blocks, 2);
        dim3 const dimBlockBlock = dim3(blocks, blocks);

        cudaMemcpyAsync(reach, graph_h->getData(), graph_h->getDataSize(), cudaMemcpyHostToDevice, cuStream);
        GPU::arrangeKernel<<<dimBlockBlock, 128, 0, cuStream>>>(_nNodes, reach, reach_);

        for (u32 i = 0; i < _nNodes; i += 128)
        {
            GPU::reachabilityArrangedKernel1<<<1, 128, 0, cuStream>>>(i, _nNodes, reach_);
            GPU::reachabilityArrangedKernel2<<<dimBlock2, 128, 0, cuStream>>>(i, _nNodes, reach_);
            GPU::reachabilityArrangedKernel3<<<dimBlockBlock, 128, 0, cuStream>>>(i, _nNodes, reach_);
        }

        GPU::transposeArrangedKernel<<<dimBlockBlock, 128, 0, cuStream>>>(_nNodes, reach_, reach_t);
        GPU::bitwiseAndKernel<<<dimBlockBlock, 128, 0, cuStream>>>(_nNodes, reach_, reach_t, reach);
        GPU::initSccKernel<<<blocks, 128, 0, cuStream>>>(_nNodes, scc_d);
        GPU::findSccArrangedKernel<<<blocks, 128, 0, cuStream>>>(_nNodes, reach, scc_d);

        cudaMemcpyAsync(scc_h->getData(), scc_d, scc_h->getDataSize(), cudaMemcpyDeviceToHost, cuStream);

    }
}