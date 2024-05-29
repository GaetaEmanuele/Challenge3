#include <iostream>
#include <functional>
#include "muparser_fun.hpp"
#include "json.hpp"
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include "JacobianSolver.hpp"
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include "chrono.hpp"
#include "convergence_test.hpp"
#include "Schwartz.hpp"
#include "write_vtk.hpp"
#include "gnuplot.hpp"
using json = nlohmann::json;


double F(const double& x,const double& y){
    return 8*M_PI*M_PI*(std::sin(M_PI*x))*(std::sin(M_PI*y));
}
double Uex(const double& x,const double& y){
    return std::sin(2*M_PI*x)*std::sin(2*M_PI*y);
}

int main (int argc, char **argv){

    MPI_Init(&argc, &argv);
    Timings::Chrono chrono;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int mpi_rank;
    MPI_Comm_rank(mpi_comm, &mpi_rank);

    int mpi_size;
    MPI_Comm_size(mpi_comm, &mpi_size);
    Eigen::RowVectorXd n(5) ;
    for(std::size_t i=0;i<5;++i){
        n(i) = std::pow(2,i+4);
    }
    if(mpi_size == 1){
        //this is the serial version//
        std::cout<<"SERIAL VERSION"<<std::endl;
        std::ifstream f("../test/data.json");
        json data = json::parse(f);
        std::string funString = data["fun"];
        unsigned int dim = std::stoi(data["n"].get<std::string>());
        std::size_t it = std::stoi(data["max_it"].get<std::string>());
        double eps = std::stod(data["Tol"].get<std::string>());
        //MuparserFun F(funString);
        edp :: JacobianSolver solver(F,it,eps,dim);
        //edp :: Schwartz solver1(F,it,eps,dim);
        edp :: Solution  result = solver.solve();
        //std::cout<<solver1.solve()<<std::endl;
        std::cout<<std::endl;
        chrono.start();
        Eigen::VectorXd xn = solver.get_nodes();
        chrono.stop();
        Eigen::VectorXd yn = xn;
        std::cout<<"The procedure requires: "<<chrono.wallTime()<<" micsec"<<std::endl;
 
        std::string relativePath = "../test/Data/serial_result.vtk";
        edp::convergence_test Test(Uex,F,n,mpi_size);
        Test.plot();
        std::string path1 = "../test/Data/serial_conv_test.txt";
        std::string outputFilePath1 = "../test/Data/sequential_err.txt";
        edp::gnuplot(path1,outputFilePath1);
        //Definitio of the targhet 
        double h = solver.get_h();
        edp :: generateVTKFile(relativePath,result,dim,dim,h,h);
    }else{
        //Parallel verion//
        //in the parallel version i'm assuming that the file json is only 
        //available for rank 0
        if(mpi_rank==0){
            std::cout<<"PARALLEL VERSION"<<std::endl;
        }
        unsigned int DIM,task,size_f;
        std::size_t max_it;
        double tollerance;
        std::string funString;
        if(mpi_rank==0){
            std::ifstream file("../test/data.json");
            json Data = json::parse(file);
            funString = Data["fun"];
            DIM = std::stoi(Data["n"].get<std::string>());
            task = std::stoi(Data["n_task"].get<std::string>());
            max_it = std::stod(Data["max_it"].get<std::string>());
            tollerance = std::stod(Data["Tol"].get<std::string>());
            size_f = funString.size();
        }
        //MPI_Bcast(&size_f, 1, MPI_INT, 0, mpi_comm);
        //MPI_Bcast(&funString,size_f,MPI_CHAR,0,mpi_comm);
        MPI_Bcast(&DIM, 1, MPI_INT, 0, mpi_comm);
        MPI_Bcast(&task, 1, MPI_INT, 0, mpi_comm);
        MPI_Bcast(&max_it, 1, MPI_UNSIGNED, 0, mpi_comm);
        MPI_Bcast(&tollerance, 1, MPI_DOUBLE, 0, mpi_comm);
        //MuparserFun F(funString);
        edp :: JacobianSolver solver2(F,max_it,tollerance,DIM,task);
        chrono.start();
        edp::Solution result = solver2.solve_in_parallel();
        chrono.stop();
        std::string relativePath = "../test/Data/parallel_result.vtk";
        edp::convergence_test Test(Uex,F,n,mpi_size,task);
        //Definition of the targhet 
        if(mpi_rank==0){
            double h = solver2.get_h();
            edp :: generateVTKFile(relativePath,result,DIM,DIM,h,h);
        }
        Test.plot();
        std::string path2 = "../test/Data/parallel_conv_test.txt";
        std::string outputFilePath2 = "../test/Data/parallel_err.txt";
        if(mpi_rank==0){
        edp:: gnuplot(path2,outputFilePath2);}
    }
    MPI_Finalize();


    return 0;
}