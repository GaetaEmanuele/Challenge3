#ifndef JACOBISOLVER_H
#define JACOBISOLVER_H
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <Eigen/Dense>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include "chrono.hpp"
#include "JacobianTraits.hpp"
namespace edp{

class JacobianSolver {
    protected:
        Fun f;
        double max_it;
        double eps;
        bool convergence = false;
        double compute_error(const Solution& res,const Solution& res1);
        unsigned int dim;
        unsigned int local_n_row = dim;
        //space discretization step
        double h = 1.0/(static_cast<double>(dim-1));
        double x0=0;
        double xN=1;
        Solution res = Solution::Zero(dim, dim);
        Solution res_loc;
        Solution Force = Solution::Zero(dim, dim);
        Solution local_Force;
        Eigen::RowVectorXd xn= Eigen::RowVectorXd::LinSpaced(dim, x0, xN);
        //number of parallel task (0 is the defual value)
        int task=1;
        virtual void split_solution();
        void join_solution();
        void perform_communications(const int& mpi_rank,const int& mpi_size,Eigen::VectorXd& row_under_sent, Eigen::VectorXd& row_over_sent,Eigen::VectorXd& row_under_receive, Eigen::VectorXd& row_over_receive);
    public:
        //the constructor that will be called by the serial version
        JacobianSolver(const Fun& f_,double max_it_,double tol,unsigned int dim_):f(f_),max_it(max_it_),eps(tol),dim(dim_){};
        //the constructor that will be called by the parallel one
        JacobianSolver(const Fun& f_,double max_it_,double tol,unsigned int dim_,int task_):f(f_),max_it(max_it_),eps(tol),dim(dim_),task(task_){};
        virtual Solution solve();
        virtual Solution solve_in_parallel();
        bool is_converged()const {return convergence;} ;
        Eigen::VectorXd get_nodes()const{return xn;};
        double get_h()const{return h;};
        Timings::Chrono chrono; //this is used by the programmer for trying to understnd 
        //what part of the solution is the bottleneck of the performance

};
}
#endif /* JACOBISOLVER_H */