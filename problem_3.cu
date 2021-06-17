#include <iostream>
#include <fstream>
#include <sstream>
#include <malloc.h>
#include <time.h>
#include <algorithm>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <cuda_runtime.h>

char input_buf[256];
std::string L_outfile, U_outfile, infile; 
int N, Num;
int *x_pos, *y_pos;
double *val;

__global__ void LU_Decompose(int *head, double *A, double *L, int *x_pos, int N, int Num){
    int total = gridDim.x * blockDim.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < N; i++){
        int h_start = head[i];
        int h_end = head[i+1];
        double uii = A[h_start];
        if(id == 0) L[h_start] = 1;
        for(int j = h_start + 1 + id; j < h_end; j += total){
            double lj = A[j]/uii;
            L[j] = lj;
            int pos = x_pos[j];
            int p = head[pos];
            for(int k = j; k < h_end; k++){
                int qos = x_pos[k];
                while(x_pos[p] != qos) p++;
                A[p] -= lj * A[k];
            }
        }
        __syncthreads();
    }
}

int main(int argc, char* argv[]){
    // handle args
    if(argc != 2){
        std::cerr<<"usage : "<<argv[0]<<" [matrix_file]"<<std::endl;
        return -1;
    }
    std::ifstream mtx_in(argv[1]);
    if(!mtx_in.is_open()){
        std::cerr<<"can not open "<<argv[1]<<std::endl;
        return -1;
    }
    infile = argv[1];
    if(infile.find_last_of("/") != infile.npos){
        infile = infile.substr(infile.find_last_of("/") + 1);
    }
    if(access("result", 4) != 0){
        mkdir("result", 0777);
    }
    L_outfile = "result/L3_" + infile;
    U_outfile = "result/U3_" + infile;

    // input matrix
    while(1){
        mtx_in.getline(input_buf, 256, '\n');
        if(input_buf[0] == '%') continue;
        std::stringstream sin;
        sin<<input_buf;
        sin>>N>>N>>Num;
        break;
    }
    x_pos = (int*)malloc(Num * sizeof(int));
    y_pos = (int*)malloc(Num * sizeof(int));
    val = (double*)malloc(Num * sizeof(double));
    for(int i = 0; i < Num; i++){
        mtx_in>>x_pos[i]>>y_pos[i]>>val[i];
        x_pos[i]--;
        y_pos[i]--;
    }
    mtx_in.close();

    // LU Decompose start

    clock_t start = clock();

    int *head = (int*)malloc((N+1) * sizeof(int));
    int A_Num = 0;

    // fill in
    std::vector<std::vector<int> > fill_in(N);
    for(int i = 0; i < Num; i++){
        fill_in[y_pos[i]].push_back(x_pos[i]);
        if(i == Num || y_pos[i] != y_pos[i+1]){
            int pos = y_pos[i];
            std::sort(fill_in[pos].begin(), fill_in[pos].end());
            std::vector<int>::iterator tail = std::unique(fill_in[pos].begin(), fill_in[pos].end());
            fill_in[pos].erase(tail, fill_in[pos].end());
            if(fill_in[pos].size() >= 3){
                int fat = fill_in[pos][1];
                for(int j = 2; j < fill_in[pos].size(); j++){
                    fill_in[fat].push_back(fill_in[pos][j]);
                }
            }
            head[pos] = A_Num;
            A_Num += fill_in[pos].size();
        }
    }
    head[N] = A_Num;

    clock_t middle = clock();
    std::cerr<<"Task C of "<<argv[1]<<" costs "<<(double)(middle-start)/CLOCKS_PER_SEC<<"s to fill in."<<std::endl;

    double* A = (double*)malloc(A_Num * sizeof(double));
    double* L = (double*)malloc(A_Num * sizeof(double));

    int *x_pos_new = (int*)malloc(A_Num * sizeof(int));

    A_Num = 0; Num = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < fill_in[i].size(); j++){
            x_pos_new[A_Num] = fill_in[i][j];
            if(fill_in[i][j] == x_pos[Num] && i == y_pos[Num]){
                A[A_Num] = val[Num++];
            }
            else{
                A[A_Num] = 0;
            }
            A_Num ++;
        }
    }

    int* head_cuda;
    double* A_cuda, *L_cuda;
    int* x_pos_new_cuda;
    cudaMalloc((void**)&head_cuda, (N+1) * sizeof(int));
    cudaMalloc((void**)&A_cuda, A_Num * sizeof(double));
    cudaMalloc((void**)&L_cuda, A_Num * sizeof(double));
    cudaMalloc((void**)&x_pos_new_cuda, A_Num * sizeof(int));

    cudaMemcpy(head_cuda, head, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(A_cuda, A, A_Num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_pos_new_cuda, x_pos_new, A_Num * sizeof(int), cudaMemcpyHostToDevice);

    int block_num = 1;
    int thread_num = 1024;

    LU_Decompose<<<block_num, thread_num>>>(head_cuda, A_cuda, L_cuda, x_pos_new_cuda, N, A_Num);

    cudaMemcpy(A, A_cuda, A_Num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(L, L_cuda, A_Num * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(head_cuda);
    cudaFree(A_cuda);
    cudaFree(L_cuda);
    cudaFree(x_pos_new_cuda);
    
    clock_t end = clock();

    // LU Decompose end

    std::cerr<<"Task C of "<<argv[1]<<" costs "<<(double)(end-start)/CLOCKS_PER_SEC<<"s."<<std::endl;

    // output L & U
    if(A_Num < 3000000){
        std::ofstream L_out(L_outfile.c_str()), U_out(U_outfile.c_str());
        L_out<<N<<" "<<N<<" "<<A_Num<<std::endl;
        for(int i = 0; i < N; i++){
            for(int j = head[i]; j < head[i+1]; j++){
                L_out<<x_pos_new[j]+1<<" "<<i+1<<" "<<L[j]<<std::endl;
            }
        }
        U_out<<N<<" "<<N<<" "<<A_Num<<std::endl;
        for(int i = 0; i < N; i++){
            for(int j = head[i]; j < head[i+1]; j++){
                U_out<<i+1<<" "<<x_pos_new[j]+1<<" "<<A[j]<<std::endl;
            }
        }
        L_out.close();
        U_out.close();
    }

    free(head);
    free(A);
    free(L);
    free(x_pos_new);

    free(x_pos);
    free(y_pos);
    free(val);

    return 0;
}