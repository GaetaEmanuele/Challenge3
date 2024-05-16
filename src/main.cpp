#include <iostream>
#include <functional>
#include "muparser_fun.hpp"
#include "json.hpp"
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include "JacobianSolver.hpp"
#include <cmath>


using json = nlohmann::json;

double F(const double& x,const double& y){
    return 8*M_PI*M_PI*(std::sin(M_PI*x))*(std::cos(M_PI*y));
}


int main (){
    std::ifstream f("../test/data.json");
    json data = json::parse(f);
    /*std::string funString = data.value("fun","");
    unsigned int dim = data.value("n","");
    unsigned int task = data.value("n_task","");
    MuparserFun f(funString);
    double it = data.value("max_it","");
    double eps = data.value("Tol","")*/
    std::string funString = data["fun"];
    unsigned int dim = std::stoi(data["n"].get<std::string>());
    unsigned int task = std::stoi(data["n_task"].get<std::string>());
    double it = std::stod(data["max_it"].get<std::string>());
    double eps = std::stod(data["Tol"].get<std::string>());
    //MuparserFun F(funString);
    edp :: JacobianSolver solver(F,it,eps,dim);
    edp :: Solution  result = solver.solve();
    Eigen::VectorXd xn = solver.get_nodes();
    Eigen::VectorXd yn = xn;
    
    std::cout << result << std::endl;
    return 0;
}