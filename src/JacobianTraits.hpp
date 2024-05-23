#ifndef JacobianTraits_HPP
#define JacobianTraits_HPP
#include <iostream>
#include <functional>
#include <Eigen/Dense>

namespace edp{
using Solution = Eigen::MatrixXd;
//wrapper of the forcing term
using Fun = std :: function<double(const double&,const double&)>;
}
#endif /* JacobianTraits_HPP */