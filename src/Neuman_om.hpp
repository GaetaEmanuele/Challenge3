#ifndef NEUMAN_OM_HPP
#define NEUMAN_OM_HPP
#include "JacobianSolver.hpp"
#include "JacobianTraits.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <Eigen/Dense>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include "chrono.hpp"
namespace edp{
class Neuman_om : public JacobianSolver {
        private:
            void impose_Neuman_bd(); 
        public:
            Neuman_om(const Fun& f_,double max_it_,double tol,unsigned int dim_):JacobianSolver(f_,max_it_,tol,dim_){};
            Solution solve() override;  
            Solution solve_in_parallel() override;
};
}
#endif //NEUMAN_OM_HPP