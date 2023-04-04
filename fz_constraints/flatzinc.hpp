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

#include <solver.hpp>
#include <fz_parser/flatzinc.h>
#include <fz_constraints/bool_array.hpp>
#include <fz_constraints/bool_bin.hpp>
#include <fz_constraints/bool_lin.hpp>
#include <fz_constraints/bool_misc.hpp>
#include <fz_constraints/int_array.hpp>
#include <fz_constraints/int_bin.hpp>
#include <fz_constraints/int_lin.hpp>
#include <fz_constraints/int_tern.hpp>

namespace Factory
{
    Constraint::Ptr makeConstraint(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
}