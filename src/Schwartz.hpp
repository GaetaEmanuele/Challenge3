#ifndef SCHWARTZ_HPP
#define SCHWARTZ_HPP
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
class Schwartz : public JacobianSolver {
        private:
            //Since in this case it is enough to apply non omogeneous 
            //boundary condition and in split_solution i initialize the local_sol
            //it is enought to put the boundary condition in it 
            //and the solve in parallel method does not change 
            Solution A = Solution::Zero(dim,dim);
            void assembly_matrix();
        public:
            Schwartz(const Fun& f_,double max_it_,double tol,unsigned int dim_):JacobianSolver(f_,max_it_,tol,dim_){};
            Solution solve() override;  
            Solution solve_in_parallel()override;
};
}
#endif //SCHWARTZ_HPP