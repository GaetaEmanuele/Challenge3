#include "Schwartz.hpp"
namespace edp{
    Solution Schwartz::solve(){
        std::size_t iter = 0;
        assembly_matrix();
        for(std::size_t i=0;i<dim;++i){
            for(std::size_t j=0;j<dim;++j){
                auto y = xn(i);
                auto x = xn(j);
                Force(i,j) = f(x,y);
            }
        }
        while(iter < max_it){
            //std::cout<<iter<<std::endl;
            Solution old_res = res;
            for(std::size_t i=1;i<dim-1;++i){
                Solution Ai = A.block(dim * (i - 1), dim * (i - 1), dim, dim);
                Eigen::VectorXd F = Force.row(i);
                Eigen::PartialPivLU<Eigen::MatrixXd> lu(Ai);
                res.row(i) = lu.solve(F);
            }
            double error = compute_error(res,old_res);
            if(convergence){
                break;
            }
            ++iter;
        }
        return res;
    }
    void Schwartz::assembly_matrix(){
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        
        int mpi_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        if(mpi_size==1){
            A = Solution :: Zero(dim*dim,dim*dim);
            double inv_h2 = 1.0 / (h * h);
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                int index = i * dim + j;

                // Coefficiente centrale
                A(index, index) = 4 * inv_h2;
            
                // Coefficiente a sinistra
                if (j > 0) {
                    A(index, index - 1) = -inv_h2;
                }
            
                // Coefficiente a destra
                if (j < dim - 1) {
                    A(index, index + 1) = -inv_h2;
                }
            
                // Coefficiente sopra
                if (i > 0) {
                    A(index, index - dim) = -inv_h2;
                }
            
                // Coefficiente sotto
                if (i < dim - 1) {
                    A(index, index + dim) = -inv_h2;
                }
            }
        }
        }else{
            if(mpi_rank==0){
                A = Solution :: Zero(dim,dim);
                for(std::size_t i=0;i<dim;++i){
                    A(i,i) = 4.0/(h*h);
                    if(i-1>0 and i+1<dim ){
                        A(i,i-1) = -1.0/(h*h);
                        A(i,i+1) = -1.0/(h*h);
                        }
                }
            }
            MPI_Bcast(A.data(), dim*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    Solution Schwartz::solve_in_parallel(){
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        
        int mpi_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        //now i call the method that will split the matrix
        split_solution();
        assembly_matrix();
        //the initialiation of the forcing term is done in parrallel with openMP
        //since each iteration is indipendent from the other
        unsigned int iter=0;
        chrono.start(); 
        while(iter < max_it){
            Solution old_res = res_loc;
            if(mpi_rank==0){
                for(std::size_t i=1;i<local_n_row;++i){
                    Eigen::VectorXd F = local_Force.row(i);
                    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
                    res_loc.row(i) = lu.solve(F);
                }
            }else if(mpi_size>2 and mpi_rank>0 and mpi_rank<(mpi_size-1)){
                for(std::size_t i=0;i<local_n_row;++i){
                    Eigen::VectorXd F = local_Force.row(i);
                    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
                    res_loc.row(i) = lu.solve(F);
                }
            }else{
                for(std::size_t i=0;i<(local_n_row-1);++i){
                    Eigen::VectorXd F = local_Force.row(i);
                    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
                    res_loc.row(i) = lu.solve(F);
            }
            }

            //work on the first row or on the last row
            double error = compute_error(res_loc,old_res);
            int global_convergence = 0;
            int convergence_int = convergence ? 1 : 0;
            
            //And with values coming from different rank
            MPI_Allreduce(&convergence_int, &global_convergence, 1, MPI_INT, MPI_LAND,MPI_COMM_WORLD);
            if(global_convergence==1){
                break;
            }
            ++iter;
        }
    chrono.stop();
    join_solution();
    return res;
    }
}