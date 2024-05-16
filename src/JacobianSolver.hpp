#ifndef JACOBISOLVER_H
#define JACOBISOLVER_H
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <Eigen/Dense>

namespace edp{
using Solution = Eigen::MatrixXd;

class JacobianSolver {
    private:
        using Fun = std :: function<double(const double&,const double&)>;
        Fun f;
        double max_it;
        double eps;
        bool convergence = false;
        double compute_error(const Solution& res,const Solution& res1);
        unsigned int dim;
        double h = 1/(static_cast<double>(dim-1));
        double x0=0;
        double xN=1;
        Eigen::VectorXd xn;
    public:
        JacobianSolver(const Fun& f_,double max_it_,double tol,unsigned int dim_):f(f_),max_it(max_it_),eps(tol),dim(dim_){};
        Solution solve();
        bool is_converged()const {return convergence;} ;
        Eigen::VectorXd get_nodes()const{return xn;};

};
}
#endif /* JACOBISOLVER_H */