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

#include <intvar.hpp>
#include <fz_parser/flatzinc.h>

class int_bin : public Constraint
{
    protected:
        var<int>::Ptr _a;
        var<int>::Ptr _b;

    public:
        int_bin(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
};

class int_bin_reif : public int_bin
{
    protected:
        var<bool>::Ptr _r;

    public:
        int_bin_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
};

class int_abs : public int_bin
{
    public:
        int_abs(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
};

class int_eq : public int_bin
{
    public:
        int_eq(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
        static void propagate(Constraint* c, var<int>::Ptr _a, var<int>::Ptr _b);
};

class int_eq_imp : public int_bin_reif
{
public:
    int_eq_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
    void post() override;
    void propagate() override;
};

class int_eq_reif : public int_bin_reif
{
    public:
        int_eq_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
};


class int_le : public int_bin
{
    public:
        int_le(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
        static void propagate(Constraint* c, var<int>::Ptr _a, var<int>::Ptr _b);
};

class int_le_imp : public int_bin_reif
{
public:
    int_le_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
    void post() override;
    void propagate() override;
};

class int_le_reif : public int_bin_reif
{
    public:
        int_le_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
};

class int_lt : public int_bin
{
    public:
        int_lt(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
        static void propagate(Constraint* c, var<int>::Ptr _a, var<int>::Ptr _b);
};

class int_lt_reif : public int_bin_reif
{
    public:
        int_lt_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
};

class int_ne : public int_bin
{
    public:
        int_ne(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
        static void propagate(Constraint* c, var<int>::Ptr _a, var<int>::Ptr _b);
};

class int_ne_reif : public int_bin_reif
{
    public:
        int_ne_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
};
