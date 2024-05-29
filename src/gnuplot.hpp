#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <sstream>
#include <cstdlib>
namespace edp
{
    


inline void gnuplot(const std::string& path1,const std::string&outputFilePath1){
    Eigen::RowVectorXd n(5) ;
    for(std::size_t i=0;i<5;++i){
        n(i) = std::pow(2,i+4);
    }
     // open the file
    std::ifstream inputFile(path1);
    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file " << path1 << std::endl;
    }

    // 
    std::vector<double> fx_values;
    double value;
    while (inputFile >> value) {
        fx_values.push_back(value);
    }
    inputFile.close();

    std::ofstream outputFile(outputFilePath1);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open file " << outputFilePath1 << std::endl;
    }

    // Scrivi i valori di x e f(x) nel file di output
    for (size_t i = 0; i < n.size(); ++i) {
        outputFile << n(i) << " " << fx_values[i] << std::endl;
    }
    outputFile.close();

    std::cout << "Data has been written to " << outputFilePath1 << " successfully";

    std::cout<<std::endl;
    }
   /* Gnuplot gp;
    gp << "plot '" + outputFilePath1 + "' with lines title 'error'\n";*/

}
// namespace edp