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

using json = nlohmann::json;


double F(const double& x,const double& y){
    return 8*M_PI*M_PI*(std::sin(M_PI*x))*(std::cos(M_PI*y));
}

int main (int argc, char **argv){

    MPI_Init(&argc, &argv);
    Timings::Chrono chrono;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int mpi_rank;
    MPI_Comm_rank(mpi_comm, &mpi_rank);

    int mpi_size;
    MPI_Comm_size(mpi_comm, &mpi_size);
    if(mpi_size == 1){
        //this is the serial version//
        std::cout<<"SERIAL VERSION"<<std::endl;
        std::ifstream f("../test/data.json");
        json data = json::parse(f);
        std::string funString = data["fun"];
        unsigned int dim = std::stoi(data["n"].get<std::string>());
        double it = std::stod(data["max_it"].get<std::string>());
        double eps = std::stod(data["Tol"].get<std::string>());
        //MuparserFun F(funString);
        edp :: JacobianSolver solver(F,it,eps,dim);
        edp :: Solution  result = solver.solve();
        chrono.start();
        Eigen::VectorXd xn = solver.get_nodes();
        chrono.stop();
        Eigen::VectorXd yn = xn;
        std::cout<<"The procedure requires: "<<chrono.wallTime()<<" micsec"<<std::endl;
        std::string relativePath = "../test/Data/serial_result.txt";

        //Definitio of the targhet 
        std::ofstream outFile(relativePath);
    
        if (outFile.is_open()) {
            //Write in the file
            outFile << result << std::endl;
            outFile.close(); // Close the file after the writing
            std::cout << "The result is stored inside: " << relativePath << std::endl;
        } else {
            std::cerr << "Error, The file cannot be opened: " << relativePath << std::endl;
        }
    }else{
        //Parallel verion//
        //in the parallel version i'm assuming that the file json is only 
        //available for rank 0
        if(mpi_rank==0){
            std::cout<<"PARALLEL VERSION"<<std::endl;
        }
        unsigned int DIM,task;
        double max_it,tollerance;
        if(mpi_rank==0){
            std::ifstream file("../test/data.json");
            json Data = json::parse(file);
            //funString = data["fun"];
            DIM = std::stoi(Data["n"].get<std::string>());
            task = std::stoi(Data["n_task"].get<std::string>());
            max_it = std::stod(Data["max_it"].get<std::string>());
            tollerance = std::stod(Data["Tol"].get<std::string>());
            //size_f = funString.size();
        }
        //MPI_Bcast(&size_f, 1, MPI_INT, 0, mpi_comm);
        //MPI_Bcast(&funString,size_f,MPI_CHAR,0,mpi_comm);
        MPI_Bcast(&DIM, 1, MPI_INT, 0, mpi_comm);
        MPI_Bcast(&task, 1, MPI_INT, 0, mpi_comm);
        MPI_Bcast(&max_it, 1, MPI_DOUBLE, 0, mpi_comm);
        MPI_Bcast(&tollerance, 1, MPI_DOUBLE, 0, mpi_comm);
    
        edp :: JacobianSolver solver2(F,max_it,tollerance,DIM,task);
        chrono.start();
        edp::Solution result = solver2.solve_in_parallel();
        chrono.stop();
        std::string relativePath = "../test/Data/parallel_result.txt";

        //Definition of the targhet 
        if(mpi_rank==0){
            std::ofstream outFile(relativePath);
            std::cout<<"The procedure requires: "<<chrono.wallTime()<<" micsec"<<std::endl;
            if (outFile.is_open()) {
                //Write in the file
                outFile << result << std::endl;
                outFile.close(); // Close the file after the writing
                std::cout << "The result is stored inside: " << relativePath << std::endl;
            } else {
                std::cerr << "Error, The file cannot be opened: " << relativePath << std::endl;
            }
            //std::cout << result << std::endl;
        }
    }
    MPI_Finalize();


    return 0;
}