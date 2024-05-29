#ifndef CONVERGENCE_TEST_H
#define CONVERGENCE_TEST_H
#include <iostream>
#include <functional>
#include <Eigen/Dense>
#include <string>
#include <cmath>
#include <fstream>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include "JacobianTraits.hpp"
#include "write_vtk.hpp"
#include "JacobianSolver.hpp"
namespace edp{
    class convergence_test{
        private:
            Fun uex;
            Fun Forcing;
            Eigen::RowVectorXd n;
            std::string path_serial;
            std::string path_parallel;
            int mpi_size=0;
            int task = 1;
            double compute_error(const int& n)const;
            double x0=0;
            double xN=1;
            Eigen::RowVectorXd xn;
        public:
            convergence_test(const Fun& uex_,const Fun& F,const Eigen::RowVectorXd& N,const int& size):uex(uex_),Forcing(F),n(N),mpi_size(size){
                path_serial = "../test/Data/serial_conv_test.txt";
                path_parallel = "../test/Data/parallel_conv_test.txt";
            };
            convergence_test(const Fun& uex_,const Fun& F,const Eigen::RowVectorXd& N,const int& size,const int& Task):uex(uex_),Forcing(F),n(N),mpi_size(size),task(Task){
                path_serial = "../test/Data/serial_conv_test.txt";
                path_parallel = "../test/Data/parallel_conv_test.txt";
            };
            //This method will save in Data the .vtx file needed for plotting
            void plot()const;
    };
}
#endif /* CONVERGENCE_TEST */