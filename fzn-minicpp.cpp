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
 * Copyright (c)  2022. by Fabio Tardivo
 */


#include <iostream>
#include <vector>
#include "solver.hpp"
#include "trailable.hpp"
#include "intvar.hpp"
#include "constraint.hpp"
#include "search.hpp"
#include <fz_parser/flatzinc.h>
#include <fz_constraints/flatzinc.hpp>
#include <cxxopts.hpp>
#include <signal.h>
#include <sys/resource.h>

#include <cuda_runtime_api.h>

var<int>::Ptr makeIntVar(CPSolver::Ptr cp, FlatZinc::IntVar& fzIntVar);
var<bool>::Ptr makeBoolVar(CPSolver::Ptr cp, FlatZinc::BoolVar& fzBoolVar);
std::function<Branches(void)> makeSearchHeuristic(CPSolver::Ptr cp, FlatZinc::SearchHeuristic& sh, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
std::vector<std::function<Branches(void)>> makeSearchCombinator(CPSolver::Ptr cp, FlatZinc::FlatZincModel* fzModel,  std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars);
Limit makeLimit(int solution_limit, int time_limit);

void printFlatZincModel(FlatZinc::FlatZincModel* fzModel);

std::function<void(int)> interruptHandler;
void signalHandler(int signal) { interruptHandler(signal); }

int main(int argc,char* argv[])
{
    // Unlimited system stack size
    rlimit limit = {RLIM_INFINITY,RLIM_INFINITY};
    setrlimit(RLIMIT_STACK, &limit);

    cudaDeviceReset();
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8u * 1024u * 1024u * 1024u);

    // Parse options
    std::string description = "Minimalistic constraint solver.";
#ifndef NDEBUG
    description.append("(Debug version)");
#else
    description.append("(Release version)");
#endif
    cxxopts::Options options_parser("minicpp", description);
    options_parser.custom_help("[Options]");
    options_parser.positional_help("<FlatZinc>");
    options_parser.add_options()
            ("a,", "Print all solutions", cxxopts::value<bool>()->default_value("false"))
            ("n,", "Stop search after 'arg' solutions", cxxopts::value<int>()->default_value("1000000000"))
            ("s,", "Print search statistics", cxxopts::value<bool>()->default_value("false"))
            ("t,", "Stop search after 'arg' ms", cxxopts::value<int>()->default_value("1000000000"))
            ("v,", "Print log messages", cxxopts::value<bool>()->default_value("false"))
            ("fzn", "FlatZinc", cxxopts::value<std::string>())
            ("h,help", "Print usage");
    options_parser.parse_positional({"fzn"});
    auto options = options_parser.parse(argc, argv);

    if((not options.count("h")) and options.count("fzn"))
    {
        //Create statistics
        SearchStatistics search_statistics;

        //Create solver
        CPSolver::Ptr cp = Factory::makeSolver();

        //Signal handling
        interruptHandler = [&](int signal)
        {
            search_statistics.setSolveTime();
            if(options["s"].as<bool>())
            {
                std::cout << search_statistics;
            }
            exit(SIGINT);
        };
        signal(SIGINT, signalHandler);

        //Parse FlatZinc
        FlatZinc::FlatZincModel * const fzModel = FlatZinc::parse(options["fzn"].as<std::string>());
        TRACE(printFlatZincModel(fzModel));

        //Create variables: no longer count the vars that are constant.
        std::vector<var<int>::Ptr> int_vars;
        size_t nbV = 0;
        for(size_t i = 0; i < fzModel->int_vars.size(); i += 1) {
           auto theVar = makeIntVar(cp, fzModel->int_vars[i]);
           nbV += !theVar->isBound();
           int_vars.push_back(theVar);
        }
        search_statistics.setIntVars(nbV);
        std::vector<var<bool>::Ptr> bool_vars;
        nbV = 0;
        for(size_t i = 0; i < fzModel->bool_vars.size(); i += 1) {
           auto theVar = makeBoolVar(cp, fzModel->bool_vars[i]);
           nbV += !theVar->isBound();
           bool_vars.push_back(theVar);
        }
        search_statistics.setBoolVars(nbV);

        //Create and post constraints
        search_statistics.setPropagators(fzModel->constraints.size());
        for(size_t i = 0; i < fzModel->constraints.size(); i += 1)
        {
            cp->post(Factory::makeConstraint(cp, fzModel->constraints[i], int_vars, bool_vars));
        }

        //Create search combinator
        std::vector<std::function<Branches(void)>> search_heuristics = makeSearchCombinator(cp, fzModel, int_vars, bool_vars);
        DFSearch search(cp, land(search_heuristics));

        std::stringstream solution;
        search.onSolution([&]()
            {
                search_statistics.incrSolutions();
                solution.str("");
                fzModel->print(solution, int_vars, bool_vars);
                solution << "----------" << std::endl;
                if (options.count("a"))
                {
                    std::cout << solution.str() << std::flush;
                }
            });
        search.onBranch([&]()
            {
                search_statistics.incrBranches();
            });
        search.onFailure([&]()
            {
                search_statistics.incrFailures();
            });

        //Start search
        search_statistics.setInitTime();

        if (fzModel->method.type == FlatZinc::Method::Type::Minimization)
        {
            var<int>::Ptr const & objVar = int_vars[fzModel->objective_variable];
            Objective::Ptr obj = Factory::minimize(objVar);
            obj->onFailure([&]()
               {
                   search_statistics.incrFalseFailures();
               });
            Limit search_limit = makeLimit(options["n"].as<int>(), options["t"].as<int>());
            search.optimize(obj, search_statistics, search_limit);
        }
        else if(fzModel->method.type == FlatZinc::Method::Type::Maximization)
        {
            var<int>::Ptr const & objVar = int_vars[fzModel->objective_variable];
            Objective::Ptr obj = Factory::maximize(objVar);
            obj->onFailure([&]()
                           {
                               search_statistics.incrFalseFailures();
                           });
            Limit search_limit = makeLimit(options["n"].as<int>(), options["t"].as<int>());
            search.optimize(obj, search_statistics, search_limit);
        }
        else if (fzModel->method.type == FlatZinc::Method::Type::Satisfaction)
        {
            int solution_limit = options.count("n") ? options["n"].as<int>() : 1;
            Limit search_limit = makeLimit(solution_limit , options["t"].as<int>());
            search.solve(search_statistics, search_limit);
        }

        // Print the solution if found
        if (not options.count("a"))
        {
            std::cout << solution.str() << std::flush;
        }

        //Print termination line
        if (search_statistics.getCompleted())
        {
            if(search_statistics.getSolutions() > 0)
            {
                std::cout <<  "==========" << std::endl;
            }
            else
            {
                std::cout << "=====UNSATISFIABLE=====" << std::endl;
            }
        }

        //Statistics printing
        search_statistics.setSolveTime();
        if(options["s"].as<bool>())
        {
            search_statistics.setPropagations(cp->getPropagations());
            std::cout << search_statistics << std::flush;
        }

        exit(EXIT_SUCCESS);
    }
    else
    {
        std::cout << options_parser.help();
        exit(EXIT_SUCCESS);
    }
}


var<int>::Ptr makeIntVar(CPSolver::Ptr cp, FlatZinc::IntVar& fzIntVar)
{
    if(fzIntVar.values.empty())    
       return Factory::makeIntVar(cp, fzIntVar.min, fzIntVar.max);
    else
       return Factory::makeIntVar(cp, fzIntVar.values);
}

var<bool>::Ptr makeBoolVar(CPSolver::Ptr cp, FlatZinc::BoolVar& fzBoolVar)
{
    if (fzBoolVar.state == FlatZinc::BoolVar::Unassigned)
    {
        return Factory::makeBoolVar(cp);
    }
    else
    {
        return Factory::makeBoolVar(cp, fzBoolVar.state == FlatZinc::BoolVar::True);
    }
}

// Variable selections
template<typename Vars, typename Var>
std::function<Var(Vars)> makeVariableSelection(FlatZinc::SearchHeuristic::VariableSelection& variable_selection)
{
    if (variable_selection == FlatZinc::SearchHeuristic::VariableSelection::first_fail)
    {
       return [](Vars vars) -> Var {return first_fail<Vars,Var>(vars);};
    }
    else if (variable_selection == FlatZinc::SearchHeuristic::VariableSelection::input_order)
    {
       return [](Vars vars)-> Var {return input_order<Vars,Var>(vars);};
    }
    else if (variable_selection == FlatZinc::SearchHeuristic::VariableSelection::smallest)
    {
       return [](Vars vars) -> Var {return smallest<Vars,Var>(vars);};
    }
    else if (variable_selection == FlatZinc::SearchHeuristic::VariableSelection::largest)
    {
        return [](Vars vars) -> Var {return largest<Vars,Var>(vars);};
    }
    else
    {
        printError("Unexpected variable selection");
        exit(EXIT_FAILURE);
    }
}

template<typename Var, typename  Val>
std::function<Branches(CPSolver::Ptr cp, Var var)> makeValueSelection(FlatZinc::SearchHeuristic::ValueSelection& value_selection)
{
    if (value_selection == FlatZinc::SearchHeuristic::ValueSelection::indomain_min)
    {
        return [](CPSolver::Ptr cp, Var var) -> Branches {return indomain_min<Var>(cp, var);};
    }
    else if (value_selection == FlatZinc::SearchHeuristic::ValueSelection::indomain_max)
    {
        return [](CPSolver::Ptr cp, Var var) -> Branches {return indomain_max<Var>(cp, var);};
    }
    else if (value_selection == FlatZinc::SearchHeuristic::ValueSelection::indomain_split)
    {
        return [](CPSolver::Ptr cp, Var var) -> Branches { return indomain_split<Var>(cp, var); };
    }
    else
    {
        printError("Unexpected value selection");
        exit(EXIT_FAILURE);
    }
}

std::function<Branches(void)> makeSearchHeuristic(CPSolver::Ptr cp, FlatZinc::SearchHeuristic& sh, std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars)
{
    if(sh.type == FlatZinc::SearchHeuristic::Type::integer)
    {
        std::vector<var<int>::Ptr> decision_variables;
        for (size_t i = 0; i < sh.decision_variables.size(); i += 1)
        {
            decision_variables.push_back(int_vars[sh.decision_variables[i]]);
        }

        auto varSel = makeVariableSelection<std::vector<var<int>::Ptr>, var<int>::Ptr>(sh.variable_selection);
        auto valSel = makeValueSelection<var<int>::Ptr, int>(sh.value_selection);
        return [=]()
        {
            return valSel(cp, varSel(decision_variables));
        };
    }
    else if (sh.type == FlatZinc::SearchHeuristic::Type::boolean)
    {
        std::vector<var<bool>::Ptr> decision_variables;
        for (size_t i = 0; i < sh.decision_variables.size(); i += 1)
        {
            decision_variables.push_back(bool_vars[sh.decision_variables[i]]);
        }

        auto varSel = makeVariableSelection<std::vector<var<bool>::Ptr>, var<bool>::Ptr>(sh.variable_selection);
        auto valSel = makeValueSelection<var<bool>::Ptr, int>(sh.value_selection);
        return [=]()
        {
            return valSel(cp, varSel(decision_variables));
        };
    }
    else
    {
        printError("Unexpected search heuristic");
        exit(EXIT_FAILURE);
    }
}

std::vector<std::function<Branches(void)>> makeSearchCombinator(CPSolver::Ptr cp, FlatZinc::FlatZincModel* fzModel,  std::vector<var<int>::Ptr>& int_vars, std::vector<var<bool>::Ptr>& bool_vars)
{
    // Initialize not decision variables as all the variables
   std::set<int> int_not_decision_vars;
   std::set<int> bool_not_decision_vars;
   for (size_t i = 0; i < fzModel->int_vars.size(); i += 1)
      int_not_decision_vars.insert(int_not_decision_vars.end(), i);
   for (size_t i = 0; i < fzModel->bool_vars.size(); i += 1)
      bool_not_decision_vars.insert(bool_not_decision_vars.end(), i);

   //Create the search heuristics
    std::vector<std::function<Branches(void)>> search_heuristics;
    for (size_t i = 0; i < fzModel->search_combinator.size(); i += 1)
    {
        auto& sh =  fzModel->search_combinator[i];
        search_heuristics.push_back(makeSearchHeuristic(cp, sh, int_vars, bool_vars));
        //std::cout << "SC[" << i << "] covers " << sh.decision_variables.size() << " variables\n";
        //Remove decision variables from not decision variables
        if(sh.type == FlatZinc::SearchHeuristic::Type::integer)
        {
            for (size_t i = 0; i < sh.decision_variables.size(); i += 1)
            {
                int_not_decision_vars.erase(sh.decision_variables[i]);
            }
        }
        else
        {
            for (size_t i = 0; i < sh.decision_variables.size(); i += 1)
            {
                bool_not_decision_vars.erase(sh.decision_variables[i]);
            }
        }

    }

    //Add the default search heuristic for the not decision variables
    FlatZinc::SearchHeuristic int_not_decision_sh(FlatZinc::SearchHeuristic::Type::integer,
                                                  int_not_decision_vars,int_vars,
                                                  FlatZinc::SearchHeuristic::VariableSelection::first_fail);
    FlatZinc::SearchHeuristic bool_not_decision_sh(FlatZinc::SearchHeuristic::Type::boolean,
                                                   bool_not_decision_vars,bool_vars,
                                                   FlatZinc::SearchHeuristic::VariableSelection::first_fail);

    //std::cout << "REM INT[" << int_not_decision_sh.decision_variables.size() << "] --> "
    //          << int_not_decision_sh.variable_selection << " | "
    //          << int_not_decision_sh.value_selection << "\n";
    //std::cout << "REM BOOL[" << bool_not_decision_sh.decision_variables.size() << "] --> "
    //          << bool_not_decision_sh.variable_selection << " | "
    //          << bool_not_decision_sh.value_selection << "\n";
    
    search_heuristics.push_back(makeSearchHeuristic(cp, int_not_decision_sh, int_vars, bool_vars));
    search_heuristics.push_back(makeSearchHeuristic(cp, bool_not_decision_sh, int_vars, bool_vars));

    return search_heuristics;
}

Limit makeLimit(int solution_limit, int time_limit)
{
    return [=](SearchStatistics const & search_statistics)
    {
        return search_statistics.getSolutions() >= solution_limit or RuntimeMonitor::elapsedSince(search_statistics.getStartTime()) >= time_limit;
    };
}

void printFlatZincModel(FlatZinc::FlatZincModel* fzModel)
{
    //Print variables
    int j = 0;
    for(size_t i = 0; i < fzModel->int_vars.size(); i += 1)
    {
        std::cout << "xi(" << j << ") "<< fzModel->int_vars[i] << std::endl;
        j += 1;
    }
    for(size_t i = 0; i < fzModel->bool_vars.size(); i += 1)
    {
        std::cout << "xb(" << j << ") " << fzModel->bool_vars[i] << std::endl;
        j += 1;
    }

    //Print constraints
    for(size_t i = 0; i < fzModel->constraints.size(); i += 1)
    {
        std::cout << "c(" << i << ") " << fzModel->constraints[i] << std::endl;
    }

}
