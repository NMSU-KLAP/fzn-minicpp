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

#include <fz_constraints/bool_bin.hpp>

bool_bin::bool_bin(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    Constraint(cp),
    _a(bool_vars[fzConstraint.vars[0]]),
    _b(bool_vars[fzConstraint.vars[1]])
{}

void bool_bin::post()
{
    _a->propagateOnBind(this);
    _b->propagateOnBind(this);
}

bool_bin_reif::bool_bin_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin(cp, fzConstraint, int_vars, bool_vars),
    _r(bool_vars[fzConstraint.vars[2]])
{}

void bool_bin_reif::post()
{
    bool_bin::post();
    _r->propagateOnBind(this);
}

bool_and_imp::bool_and_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_and_imp::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_and_imp::propagate()
{
    //Semantic: r -> a /\ b
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation: r <- a /\ b
    if (aMax < bMin or bMax < aMin)
    {
        _r->assign(false);
    }

    //Propagation: r -> a /\ b
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            _a->assign(true);
            _b->assign(true);
        }
        else
        {
            setActive(false);
        }
    }
}


bool_and_reif::bool_and_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
        bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_and_reif::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_and_reif::propagate()
{
    //Semantic: a /\ b <-> r
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation: a /\ b -> r
    if (aMin == 1 and bMin == 1)
    {
        _r->assign(true);
    }
    else if (aMax < bMin or bMax < aMin)
    {
        _r->assign(false);
    }

    //Propagation: a /\ b <- r
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            _a->assign(true);
            _b->assign(true);
        }
        else
        {
            if (aMin == 1)
            {
                _b->assign(false);
            }
            else if (bMin == 1)
            {
                _a->assign(false);
            }
        }
    }
}


bool_eq::bool_eq(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_eq::post()
{
    bool_bin::post();
    propagate();
}

void bool_eq::propagate()
{
    propagate(this, _a, _b);
}

void bool_eq::propagate(Constraint* c, var<bool>::Ptr _a, var<bool>::Ptr _b)
{
    //Semantic: a = b
    int min = std::max(_a->min(), _b->min());
    int max = std::min(_a->max(), _b->max());

    //Propagation: a -> b
    _b->updateBounds(min, max);

    //Propagation: b -> a
    _a->updateBounds(min, max);
}


bool_eq_imp::bool_eq_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
        bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_eq_imp::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_eq_imp::propagate()
{
    //Semantic: r -> a = b
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation: r <- a = b
    if (aMax < bMin or bMax < aMin)
    {
        _r->assign(false);
    }

    //Propagation: r -> a = b
    else if (_r->isBound())
    {
        if (_r->isTrue())
        {
            bool_eq::propagate(this, _a, _b);
        }
        else
        {
            setActive(false);
        }
    }
}

bool_eq_reif::bool_eq_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_eq_reif::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_eq_reif::propagate()
{
    //Semantic: a = b <-> r
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation: a = b -> r
    if (aMin == aMax and bMin == bMax and aMin == bMin)
    {
        _r->assign(true);
    }
    else if (aMax < bMin or bMax < aMin)
    {
        _r->assign(false);
    }

    //Propagation: a = b <- r
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            bool_eq::propagate(this, _a, _b);
        }
        else
        {
            bool_not::propagate(this, _a, _b);
        }
    }
}

bool_le::bool_le(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_le::post()
{
    bool_bin::post();
    propagate();
}

void bool_le::propagate()
{
    propagate(this, _a, _b);
}

void bool_le::propagate(Constraint* c, var<bool>::Ptr _a, var<bool>::Ptr _b)
{
    //Semantic: a <= b
    int aMin = _a->min();
    int bMax = _b->max();

    //Propagation: a <- b
    _a->removeAbove(bMax);

    //Propagation: a -> b
    _b->removeBelow(aMin);
}

bool_le_imp::bool_le_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
        bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_le_imp::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_le_imp::propagate()
{
    //Semantic: r -> a <= b
    int aMin = _a->min();
    int bMax = _b->max();

    //Propagation: r <- a <= b
    if (bMax < aMin)
    {
        _r->assign(false);
    }

    //Propagation: r -> a <= b
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            bool_le::propagate(this, _a, _b);
        }
        else
        {
            setActive(false);
        }
    }
}

bool_le_reif::bool_le_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_le_reif::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_le_reif::propagate()
{
    //Semantic: a <= b <-> r
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation: a <= b -> r
    if (aMax <= bMin)
    {
        _r->assign(true);
    }
    else if (bMax < aMin)
    {
        _r->assign(false);
    }

    //Propagation: a <= b <- r
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            bool_le::propagate(this, _a, _b);
        }
        else
        {
            bool_lt::propagate(this, _b, _a);
        }
    }
}

bool_lt::bool_lt(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_lt::post()
{
    bool_bin::post();
    propagate();
}

void bool_lt::propagate()
{
    propagate(this, _a, _b);
}

void bool_lt::propagate(Constraint* c, var<bool>::Ptr _a, var<bool>::Ptr _b)
{
    //Semantic: a < b
    int aMin = _a->min();
    int bMax = _b->max();

    //Propagation: b -> a
    _a->removeAbove(bMax - 1);

    //Propagation: a -> b
    _b->removeBelow(aMin + 1);
}


bool_lt_imp::bool_lt_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
        bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_lt_imp::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_lt_imp::propagate()
{
    //Semantic: r -> a < b
    int aMin = _a->min();
    int bMax = _b->max();

    //Propagation: r <- a < b
    if (bMax <= aMin)
    {
        _r->assign(false);
    }

    //Propagation: r -> a < b
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            bool_lt::propagate(this, _a, _b);
        }
        else
        {
            setActive(false);
        }
    }
}

bool_lt_reif::bool_lt_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_lt_reif::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_lt_reif::propagate()
{
    //Semantic: a < b <-> r
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation: a < b -> r
    if (aMax < bMin)
    {
        _r->assign(true);
    }
    else if (bMax <= aMin)
    {
        _r->assign(false);
    }

    //Propagation: a < b <- r
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            bool_lt::propagate(this, _a, _b);
        }
        else
        {
            bool_le::propagate(this, _b, _a);
        }
    }
}

bool_not::bool_not(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_not::post()
{
    bool_bin::post();
    propagate();
}

void bool_not::propagate()
{
    propagate(this, _a, _b);
}

void bool_not::propagate(Constraint* c, var<bool>::Ptr _a, var<bool>::Ptr _b)
{
    //Semantic: a != b
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation: a -> b
    if(aMin == aMax)
    {
        _b->remove(aMin);
    }

    //Propagation: b -> a
    if (bMin == bMax)
    {
        _a->remove(bMin);
    }
}


bool_or_imp::bool_or_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
        bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_or_imp::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_or_imp::propagate()
{
    //Semantic: r -> a \/ b
    int aMax = _a->max();
    int bMax = _b->max();

    //Propagation: r <- a \/ b
    if (aMax == 0 and bMax == 0)
    {
        _r->assign(false);
    }

    //Propagation: r -> a \/ b
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            if (aMax == 0)
            {
                _b->assign(true);
            }
            else if (bMax == 0)
            {
                _a->assign(true);
            }
        }
        else
        {
            setActive(false);
        }
    }
}

bool_or_reif::bool_or_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_or_reif::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_or_reif::propagate()
{
    //Semantic: a \/ b <-> r
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation: a \/ b -> r
    if (aMin == 1 or bMin == 1)
    {
        _r->assign(true);
    }
    else if (aMax == 0 and bMax == 0)
    {
        _r->assign(false);
    }

    //Propagation: a \/ b <- r
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            if (aMax == 0)
            {
                _b->assign(true);
            }
            else if (bMax == 0)
            {
                _a->assign(true);
            }
        }
        else
        {
            _a->assign(false);
            _b->assign(false);
        }
    }
}

bool_xor::bool_xor(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_xor::post()
{
    bool_bin::post();
    propagate();
}

void bool_xor::propagate()
{
    bool_not::propagate(this, _a, _b);
}


bool_xor_imp::bool_xor_imp(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
        bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_xor_imp::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_xor_imp::propagate()
{
    //Semantic: r -> a + b
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation: r <- a + b
    if (aMin == aMax and bMin == bMax and aMin == bMin)
    {
        _r->assign(false);
    }

    //Propagation: r -> a + b
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            bool_not::propagate(this, _a, _b);
        }
        else
        {
            setActive(false);
        }
    }
}

bool_xor_reif::bool_xor_reif(CPSolver::Ptr cp, FlatZinc::Constraint& fzConstraint, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars) :
    bool_bin_reif(cp, fzConstraint, int_vars, bool_vars)
{}

void bool_xor_reif::post()
{
    bool_bin_reif::post();
    propagate();
}

void bool_xor_reif::propagate()
{
    //Semantic: a + b <-> r
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation: a + b -> r
    if (aMax < bMin or bMax < aMin)
    {
        _r->assign(true);
    }
    else if (aMin == aMax and bMin == bMax and aMin == bMin)
    {
        _r->assign(false);
    }

    //Propagation: a + b <- r
    if (_r->isBound())
    {
        if (_r->isTrue())
        {
            bool_not::propagate(this, _a, _b);
        }
        else
        {
            bool_eq::propagate(this, _a, _b);
        }
    }
}

