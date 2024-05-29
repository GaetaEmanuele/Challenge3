#include "convergence_test.hpp"

namespace edp{
    void convergence_test :: plot()const {
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        Eigen::RowVectorXd error(n.size());
        for(std::size_t i=0;i< n.size();++i){
            double h = 1.0/(static_cast<double>(n(i)-1));
            double err = compute_error(n(i));
            error(i) = std::sqrt(h* err);
            if(mpi_rank==0){
                std::cout<<"iter: "<<i<<std::endl;}   
            }
        if(mpi_size ==1){
            std::ofstream outFile(path_serial);
            if (outFile.is_open()) {
                //Write in the file
                outFile << error << std::endl;
                outFile.close(); // Close the file after the writing
                std::cout << "The result is stored inside: " << path_serial << std::endl;
            } else {
                std::cerr << "Error, The file cannot be opened: " << path_serial << std::endl;
            }
        }else if(mpi_rank==0){
            std::ofstream outFile(path_parallel);
            if (outFile.is_open()) {
                //Write in the file
                outFile << error << std::endl;
                outFile.close(); // Close the file after the writing
                std::cout << "The result is stored inside: " << path_parallel << std::endl;
            } else {
                std::cerr << "Error, The file cannot be opened: " << path_parallel << std::endl;
            }
        }
    }
    double convergence_test :: compute_error(const int& n)const{
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        if(mpi_size ==1){
             JacobianSolver solver(Forcing,1e6,1e-3,n);
             Eigen::RowVectorXd xn = Eigen::VectorXd::LinSpaced(n, 0.0, 1.0);
             Solution U_app = solver.solve();
             //Solution Uex = Solution::Zero(n,n);
             double sum=0.0;
             for(std::size_t i=1;i<n-1;++i){
                for(std::size_t j=1;j<n-1;++j){
                    double ue = uex(xn(j),xn(i)); 
                    sum += (U_app(i,j)-ue)*(U_app(i,j)-ue);
                }
             }
             return sum;
        }else{
            std::vector<int> count_recv(mpi_size);
            std::vector<int> displacements(mpi_size);
            //rank 0 calculate the size of chunks 
            //for splitting the work in omegeneous way;
            if(mpi_rank ==0){
                auto chunk = n/mpi_size;
                auto rest  = n%mpi_size;
                #pragma omp parallel for num_threads(task)
                for (auto i=0;i<mpi_size;++i){
                    count_recv[i]= i<rest? chunk+1: chunk;
                    }
                displacements[0]=0;
                for (auto i=1;i<mpi_size;++i){
                    displacements[i]=displacements[i-1]+count_recv[i-1];
                    }
            }

            int local_n_row=0;
            MPI_Scatter(count_recv.data(),1,MPI_INT, &local_n_row,1,MPI_INT,0,MPI_COMM_WORLD);
            MPI_Bcast(displacements.data(), mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
            int j = displacements[mpi_rank];
            JacobianSolver solver(Forcing,1e6,1e-3,n,task);
            Eigen::RowVectorXd xn = Eigen::VectorXd::LinSpaced(n, 0.0, 1.0);
            /*Eigen::RowVectorXd xn_loc(local_n_row);
            for(std::size_t i=0;i<local_n_row;++i){
                xn_loc(i)=xn(j);
                ++j;
            }*/
            Solution res = solver.solve();
            MPI_Bcast(res.data(), n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            j = displacements[mpi_rank];
            double global_err=0.0;
            double local_err=0.0;
            #pragma omp parallel for num_threads(task) collapse(2)
            for(std::size_t i=j;i<(j+local_n_row);++i){
                for(std::size_t jj=0;jj<n;++jj){
                    auto x = xn(jj);
                    auto y = xn(i-j);
                    double U = res(i,jj);
                    double U_ex = uex(x,y);
                    local_err += (U-U_ex)*(U-U_ex);
                }
             }
            MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            return global_err;
        }
    
    }

}