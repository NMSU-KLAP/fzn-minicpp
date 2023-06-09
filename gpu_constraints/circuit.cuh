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

#include "constraint.hpp"

class CircuitGPU : public Circuit
{
    public:
        template <class Vec> CircuitGPU(Vec const & x, int offset = 0) : Circuit(x, offset) { setPriority(CLOW);};
    protected:
        void postAllDifferent() override;
};
