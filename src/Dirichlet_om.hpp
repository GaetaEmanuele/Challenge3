#ifndef DIRCHLET_OM_HPP
#define DIRICHLET_OM_HPP
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
class Dirichlet_om : public JacobianSolver {
        private:
            Fun bd_cond;
            //Since in this case it is enough to apply non omogeneous 
            //boundary condition and in split_solution i initialize the local_sol
            //it is enought to put the boundary condition in it 
            //and the solve in parallel method does not change 
            void split_solution() override;
        public:
            Dirichlet_om(const Fun& f_,double max_it_,double tol,unsigned int dim_,const Fun& cond):JacobianSolver(f_,max_it_,tol,dim_),bd_cond(cond){};
            Solution solve() override;  
};
}
#endif //DIRICHELT_OM_HPP