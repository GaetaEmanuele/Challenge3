#include "Neuman_om.hpp"

namespace edp{
    Solution Neuman_om::solve(){
        std::size_t iter = 0;
        for(std::size_t i=0;i<dim;++i){
            for(std::size_t j=0;j<dim;++j){
                auto y = xn(i);
                auto x = xn(j);
                Force(i,j) = f(x,y);
            }
        }
        while(iter < max_it){
            Solution old_res = res;
            for(std::size_t i=1;i<dim-1;++i){
                for(std::size_t j=1;j<dim-1;++j){
                    res(i,j) = 0.25 * (res(i-1,j)+res(i+1,j)+res(i,j-1)+res(i,j+1)+h*h*Force(i,j));
                }
            }
            impose_Neuman_bd();
            double error = compute_error(res,old_res);
            if(convergence){
                break;
            }
            ++iter;
        }
        return res;
    }
    void Neuman_om::impose_Neuman_bd(){
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

        int mpi_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        if(mpi_size==1){
            res.row(0) = res.row(1);        // upper boundary
            res.row(dim-1) = res.row(dim-2);      // lower boundary

            // left and right boundary
            res.col(0) = res.col(1);        
            res.col(dim-1) = res.col(dim-2); 
        }else{
            if(mpi_rank==0){
                res_loc.row(0) = res_loc.row(1);
            }
            if(mpi_rank == mpi_size-1){
                res_loc.row(dim-1) = res_loc.row(dim-2); 
            }
                        // left and right boundary
            res_loc.col(0) = res_loc.col(1);        
            res_loc.col(dim-1) = res_loc.col(dim-2); 
        }
    }
    Solution Neuman_om::solve_in_parallel(){
        //since the mpi_rank and mpi_size are out of the scope of the class (they are in the main scope)
        //i need to reinitialize them (the same in the following method)
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        
        int mpi_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        //now i call the method that will split the matrix
        split_solution();
        //the initialiation of the forcing term is done in parrallel with openMP
        //since each iteration is indipendent from the other
        unsigned int iter=0;
        chrono.start(); 
        while(iter < max_it){
            Solution old_res = res_loc;
            Eigen::VectorXd row_under_sent(dim);
            Eigen::VectorXd row_over_sent(dim);
            Eigen::VectorXd row_under_recive(dim);
            Eigen::VectorXd row_over_recive(dim);
            //MPI_Request request1;
            //MPI_Request request2;
            perform_communications(mpi_rank,mpi_size,row_under_sent,row_over_sent,row_under_recive,row_over_recive);
            if(mpi_rank==0){
                for(std::size_t i=1;i<local_n_row;++i){
                    for(std::size_t j=1;j<dim-1;++j){
                        if(i==local_n_row-1){
                            res_loc(i,j) = 0.25 * (res_loc(i-1,j)+row_over_recive(j)+res_loc(i,j-1)+res_loc(i,j+1)+h*h*local_Force(i,j));
                        }else{
                        res_loc(i,j) = 0.25 * (res_loc(i-1,j)+res_loc(i+1,j)+res_loc(i,j-1)+res_loc(i,j+1)+h*h*local_Force(i,j));
                    }
                    }
                }
            }else if(mpi_size>2 and mpi_rank>0 and mpi_rank<(mpi_size-1)){
                for(std::size_t i=0;i<local_n_row;++i){
                    for(std::size_t j=1;j<dim-1;++j){
                        if(i==0){
                            res_loc(i,j) = 0.25 * (row_under_recive(j)+res_loc(i+1,j)+res_loc(i,j-1)+res_loc(i,j+1)+h*h*local_Force(i,j));
                        }
                        //since if the local_n_row is <=2 
                        //i must use always the 2 vector obtained by the comunication
                        if(i>=1 and i<(local_n_row-1)){
                            res_loc(i,j) = 0.25 * (res_loc(i-1,j)+res_loc(i+1,j)+res_loc(i,j-1)+res_loc(i,j+1)+h*h*local_Force(i,j));
                        }
                        if(i==(local_n_row-1)){
                            res_loc(i,j) = 0.25 * (res_loc(i-1,j)+row_over_recive(j)+res_loc(i,j-1)+res_loc(i,j+1)+h*h*local_Force(i,j));
                        }
                    }
                }
            }else{
                for(std::size_t i=0;i<(local_n_row-1);++i){
                    for(std::size_t j=1;j<dim-1;++j){
                        if(i==0){
                            res_loc(i,j) = 0.25 * (row_under_recive(j)+res_loc(i+1,j)+res_loc(i,j-1)+res_loc(i,j+1)+h*h*local_Force(i,j));
                        }else{
                        res_loc(i,j) = 0.25 * (res_loc(i-1,j)+res_loc(i+1,j)+res_loc(i,j-1)+res_loc(i,j+1)+h*h*local_Force(i,j));
                }
                }
            }
            }

            //work on the first row or on the last row
            impose_Neuman_bd();
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