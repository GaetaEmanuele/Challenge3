#include "JacobianSolver.hpp"

namespace edp{
    Solution JacobianSolver :: solve(){
        std::size_t iter = 0;
        for(int i=0;i<dim;++i){
            double y = xn(i);
            for(int j=0;j<dim;++j){
                    double x = xn(j);
                    Force(i,j) = f(x,y);
            }
        }

        while(iter < max_it){
            Solution old_res = res;
            for(std::size_t i=1;i<dim-1;++i){
                for(std::size_t j=1;j<dim-1;++j){
                    res(i,j) = 0.25 * (old_res(i-1,j)+old_res(i+1,j)+old_res(i,j-1)+old_res(i,j+1)+h*h*Force(i,j));
                }
            }
            double error = compute_error(res,old_res);
            if(convergence){
                break;
            }
            ++iter;
        }
        return res;
    }

 inline   double JacobianSolver :: compute_error(const Solution& res,const Solution& old_res){
        double error=0.0;

        #pragma omp parallel for num_threads(task) collapse(2)
        for(std::size_t i=0;i<local_n_row;++i){
            for(std::size_t j=0;j<dim;++j){
                double t = (res(i,j)-old_res(i,j))*(res(i,j)-old_res(i,j));
                error += t;
            }
        }
        error = std::sqrt(h*error);
        if(error < eps){
            convergence = true;
        }
        return error;
    }

    Solution JacobianSolver:: solve_in_parallel(){
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

    void JacobianSolver::split_solution(){
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

        int mpi_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        //split the number of rows of the matrix for do the partitioning of 
        //the work;
        std::vector<int> count_recv(mpi_size);
        std::vector<int> displacements(mpi_size);
        //rank 0 calculate the size of chunks 
        //for splitting the work in omegeneous way;
        if(mpi_rank ==0){
            auto chunk = dim/mpi_size;
            auto rest  = dim%mpi_size;
            #pragma omp parallel for num_threads(task)
            for (auto i=0;i<mpi_size;++i){
                count_recv[i]= i<rest? chunk+1: chunk;
                }
            displacements[0]=0;
            for (auto i=1;i<mpi_size;++i){
                displacements[i]=displacements[i-1]+count_recv[i-1];
                }
        }
        MPI_Scatter(count_recv.data(),1,MPI_INT, &local_n_row,1,MPI_INT,0,MPI_COMM_WORLD);
        res_loc = Solution::Zero(local_n_row,dim);
        std::size_t j=0;
        MPI_Scatter(displacements.data(),1,MPI_INT, &j,1,MPI_INT,0,MPI_COMM_WORLD);
        Eigen::RowVectorXd xn_loc(local_n_row);
        for(std::size_t i=0;i<local_n_row;++i){
            xn_loc(i)=xn(j);
            ++j;
        }
               
        local_Force = Solution::Zero(local_n_row,dim);

        #pragma omp parallel for num_threads(task) collapse(2)
        for(std::size_t i=0;i<local_n_row;++i){
            for(std::size_t j=0;j<dim;++j){
                auto x = xn(j);
                auto y=xn_loc(i);
                local_Force(i,j) = f(x,y); 
            }
        }
    }

inline  void JacobianSolver:: join_solution(){
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    auto nrow = 0;
    //MPI_Request request1;
    //MPI_Request request2;
    if(mpi_rank==0){
        res.block(0, 0, local_n_row, dim) = res_loc;
        nrow +=local_n_row;
        for(std::size_t i=1;i<mpi_size;++i){
            int local_size=0;
            //MPI_Irecv(&local_size, 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE,&request1);
            MPI_Recv(&local_size, 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Solution loc_res(local_size,dim);
            //MPI_Irecv(loc_res.data(), local_size*dim, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE,&request2);
            MPI_Recv(loc_res.data(), local_size*dim, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            res.block(nrow, 0, local_size, dim) = loc_res;
            nrow += local_size;
        }
    }else{
            //MPI_Isend(&local_n_row,1,MPI_INT , 0, mpi_rank, MPI_COMM_WORLD,&request1);
            MPI_Send(&local_n_row,1,MPI_INT , 0, mpi_rank, MPI_COMM_WORLD);   
            //MPI_Isend(&local_n_row,1,MPI_INT , 0, mpi_rank, MPI_COMM_WORLD,&request1);
            MPI_Send(res_loc.data(),local_n_row*dim,MPI_DOUBLE , 0, mpi_rank, MPI_COMM_WORLD);
    }
    } 

inline void JacobianSolver::perform_communications(const int& mpi_rank,const int& mpi_size,Eigen::VectorXd& row_under_sent, Eigen::VectorXd& row_over_sent,
                                             Eigen::VectorXd& row_under_receive, Eigen::VectorXd& row_over_receive) {
//Since this function will be called in a iterative loop 
//i will pass the information of rank and size since this will reduce the time needed 
    if (mpi_rank == 0) {
        row_over_sent = res_loc.row(local_n_row - 1);
        MPI_Send(row_over_sent.data(), dim, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(row_over_receive.data(), dim, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (mpi_size > 2 && mpi_rank < (mpi_size - 1)) {
        MPI_Recv(row_under_receive.data(), dim, MPI_DOUBLE, mpi_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        row_under_sent = res_loc.row(0);
        MPI_Send(row_under_sent.data(), dim, MPI_DOUBLE, mpi_rank - 1, 1, MPI_COMM_WORLD);
        row_over_sent = res_loc.row(local_n_row - 1);
        MPI_Send(row_over_sent.data(), dim, MPI_DOUBLE, mpi_rank + 1, 0, MPI_COMM_WORLD);
        MPI_Recv(row_over_receive.data(), dim, MPI_DOUBLE, mpi_rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        row_under_sent = res_loc.row(0);
        MPI_Recv(row_under_receive.data(), dim, MPI_DOUBLE, mpi_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(row_under_sent.data(), dim, MPI_DOUBLE, mpi_rank - 1, 1, MPI_COMM_WORLD);
    }
}


 }

