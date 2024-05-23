#include "Dirichlet_om.hpp"

namespace edp{
    Solution Dirichlet_om:: solve(){
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
            for(std::size_t i=0;i<dim;++i){
                for(std::size_t j=0;j<dim;++j){
                    if(i==0 or j==0){
                        auto y = xn(i);
                        auto x = xn(j);
                        res(i,j) = bd_cond(x,y); 
                    }else{
                        res(i,j) = 0.25 * (res(i-1,j)+res(i+1,j)+res(i,j-1)+res(i,j+1)+h*h*Force(i,j));
                    }
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

    void Dirichlet_om:: split_solution(){
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

        //So in ths loop i'm also imposing the boundary condition
        //while in the base class this part of the method was used only for splitting
        //the force bwtween the different rank
        #pragma omp parallel for num_threads(task) collapse(2)
        for(std::size_t i=0;i<local_n_row;++i){
            for(std::size_t j=0;j<dim;++j){
                auto x = xn(j);
                auto y=xn_loc(i);
                local_Force(i,j) = f(x,y); 
                if(i==0 or j==0){
                    res_loc(i,j) = bd_cond(x,y);
                }
            }
        }
    }
}