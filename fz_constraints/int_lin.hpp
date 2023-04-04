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

class int_lin : public Constraint
{
    protected:
        std::vector<int> _as_pos;
        std::vector<int> _as_neg;
        std::vector<var<int>::Ptr> _bs_pos;
        std::vector<var<int>::Ptr> _bs_neg;
        int _c;
        int _sumMin;
        int _sumMax;
        int _posNotBoundCount;
        int _negNotBoundCount;
        int _posNotBoundIdx;
        int _negNotBoundIdx;

    public:
        int_lin(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        static void calSumMinMax(int_lin* il);
        void post() override;

    friend class int_lin_eq;
    friend class int_lin_ge;
    friend class int_lin_le;
    friend class int_lin_ne;
};

class int_lin_reif : public int_lin
{
    protected:
        var<bool>::Ptr _r;

    public:
        int_lin_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
};

class int_lin_eq : public int_lin
{
    public:
        int_lin_eq(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
        static void propagate(int_lin* il);
};

class int_lin_eq_imp : public int_lin_reif
{
    public:
        int_lin_eq_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
};

class int_lin_eq_reif : public int_lin_reif
{
    public:
        int_lin_eq_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
};

class int_lin_ge
{
    public:
        static void propagate(int_lin* il, int c);
};

class int_lin_le : public int_lin
{
    public:
        int_lin_le(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
        static void propagate(int_lin* il);
};

class int_lin_le_imp : public int_lin_reif
{
    public:
        int_lin_le_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
};

class int_lin_le_reif : public int_lin_reif
{
    public:
        int_lin_le_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
};

class int_lin_ne : public int_lin
{
    public:
        int_lin_ne(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
        static void propagate(int_lin* il);
};

class int_lin_ne_imp : public int_lin_reif
{
    public:
        int_lin_ne_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
        void post() override;
        void propagate() override;
};

class int_lin_ne_reif : public int_lin_reif
{
public:
    int_lin_ne_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
    void post() override;
    void propagate() override;
};
