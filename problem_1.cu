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

char input_buf[256];
std::string L_outfile, U_outfile, infile; 
int N, Num;
int *x_pos, *y_pos;
double *val, *A, *L, *U;
const double eps = 0;

void LU_Decompose(double *A, double *L, double *U, int N){
    for(int i = 0; i < N; i++){
        L[i*N + i] = 1;
        for(int j = i; j < N; j++){
            double SU = 0;
            for(int k = 0; k < i; k++){
                SU += L[i*N + k] * U[k*N + j];
            }
            U[i*N + j] = A[i*N + j] - SU;
            if(j + 1 < N){
                double SL = 0;
                for(int k = 0; k < i; k++){
                    SL += L[(j+1)*N + k] * U[k*N + i];
                }
                L[(j+1)*N + i] = (A[(j+1)*N + i] - SL)/U[i*N + i];
            }
        }
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
    L_outfile = "result/L1_" + infile;
    U_outfile = "result/U1_" + infile;

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

    A = (double*)malloc(N * N * sizeof(double));
    L = (double*)malloc(N * N * sizeof(double));
    U = (double*)malloc(N * N * sizeof(double));

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            A[i*N + j] = 0;
            L[i*N + j] = 0;
            U[i*N + j] = 0;
        }
    }
    for(int i = 0; i < Num; i++){
        A[x_pos[i]*N + y_pos[i]] = val[i];
        A[y_pos[i]*N + x_pos[i]] = val[i];
    }

    // LU Decompose start

    clock_t start = clock();

    LU_Decompose(A, L, U, N);
    
    clock_t end = clock();

    // LU Decompose end

    std::cerr<<"Task A of "<<argv[1]<<" costs "<<(double)(end-start)/CLOCKS_PER_SEC<<"s."<<std::endl;

    // output L & U
    std::ofstream L_out(L_outfile.c_str()), U_out(U_outfile.c_str());
    int L_Num = 0, U_Num = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(abs(L[i*N + j]) > eps) L_Num ++;
        }
    }
    L_out<<N<<" "<<N<<" "<<L_Num<<std::endl;
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            if(abs(L[i*N + j]) > eps){
                L_out<<i+1<<" "<<j+1<<" "<<L[i*N + j]<<std::endl;
            }
        }
    }
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(abs(U[i*N + j]) > eps) U_Num ++;
        }
    }
    U_out<<N<<" "<<N<<" "<<U_Num<<std::endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(abs(U[i*N + j]) > eps){
                U_out<<i+1<<" "<<j+1<<" "<<U[i*N + j]<<std::endl;
            }
        }
    }
    L_out.close();
    U_out.close();

    free(A);
    free(L);
    free(U);

    free(x_pos);
    free(y_pos);
    free(val);

    return 0;
}