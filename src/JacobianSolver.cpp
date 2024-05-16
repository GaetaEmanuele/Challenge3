#include "JacobianSolver.hpp"

namespace edp{
    Solution JacobianSolver :: solve(){
        Solution res = Solution::Zero(dim, dim);
        std::size_t iter = 0;
        xn = Eigen::VectorXd::LinSpaced(dim, x0, xN);
        Solution Force = res;
        for(std::size_t i=0;i<dim;++i){
            for(std::size_t j=0;j<dim;++j){
                auto y = xn(i);
                auto x = xn(j);
                Force(i,j) = f(x,y);
            }
        }
        while(iter < max_it){
            Solution old_res = res;
            for(std::size_t i=1;i<dim-1;++i){
                for(std::size_t j=1;j<dim-1;++j){
                    res(i,j) = 0.25 * (res(i-1,j)+res(i+1,j)+res(i,j-1)+res(i,j+1)+h*h*Force(i,j));
                }
            }
            double error = compute_error(res,old_res);
            if(convergence){
                break;
            }
            ++iter;
        }
        return res;
    }

    double JacobianSolver :: compute_error(const Solution& res,const Solution& old_res){
        double error=0.0;
        for(std::size_t i=0;i<dim;++i){
            for(std::size_t j=0;j<dim;++j){
                error += (res(i,j)-old_res(i,j))*(res(i,j)-old_res(i,j));
            }
        }
        error = std::sqrt(h*error);
        if(error < eps){
            convergence = true;
        }
        return error;
    }
}