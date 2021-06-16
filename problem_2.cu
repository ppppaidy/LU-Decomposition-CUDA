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
#include <unordered_map>

char input_buf[256];
std::string L_outfile, U_outfile, infile; 
int N, Num;
int *x_pos, *y_pos;
double *val;

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
    L_outfile = "result/L2_" + infile;
    U_outfile = "result/U2_" + infile;

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

    std::vector<std::unordered_map<int, double> > L_val(N);
    std::vector<std::unordered_map<int, double> > U_val(N);
    for(int i = 0; i < Num; i++){
        U_val[y_pos[i]][x_pos[i]] += val[i];
        L_val[y_pos[i]][x_pos[i]] += val[i];
        if(i == Num-1 || y_pos[i] != y_pos[i+1]){
            double uii = U_val[y_pos[i]][y_pos[i]];
            for(std::unordered_map<int, double>::iterator it = L_val[y_pos[i]].begin(); it != L_val[y_pos[i]].end(); it++){
                it->second /= uii;
            }
            L_val[y_pos[i]][y_pos[i]] = 1;
        }
        if(i < Num-1 && y_pos[i] != y_pos[i+1]){
            for(std::unordered_map<int, double>::iterator it1 = L_val[y_pos[i]].begin(); it1 != L_val[y_pos[i]].end(); it1++){
                for(std::unordered_map<int, double>::iterator it2 = U_val[y_pos[i]].begin(); it2 != U_val[y_pos[i]].end(); it2++){
                    if(it1->first != y_pos[i] && it2->first != y_pos[i]){
                        if(it1->first <= it2->first){
                            U_val[it1->first][it2->first] -= it1->second * it2->second;
                        }
                        else{
                            L_val[it2->first][it1->first] -= it1->second * it2->second;
                        }
                    }
                }
            }
        }
    }
    
    clock_t end = clock();

    // LU Decompose end

    std::cerr<<"Task B of "<<argv[1]<<" costs "<<(double)(end-start)/CLOCKS_PER_SEC<<"s."<<std::endl;

    // output L & U
    std::ofstream L_out(L_outfile.c_str()), U_out(U_outfile.c_str());
    int L_Num = 0, U_Num = 0;
    for(int i = 0; i < N; i++){
        for(std::unordered_map<int, double>::iterator it = U_val[i].begin(); it != U_val[i].end(); it++){
            U_Num ++;
        }
    }
    L_Num = U_Num;
    U_out<<N<<" "<<N<<" "<<U_Num<<std::endl;
    for(int i = 0; i < N; i++){
        for(std::unordered_map<int, double>::iterator it = U_val[i].begin(); it != U_val[i].end(); it++){
            U_out<<i+1<<" "<<it->first+1<<" "<<it->second<<std::endl;
        }
    }
    L_out<<N<<" "<<N<<" "<<L_Num<<std::endl;
    for(int i = 0; i < N; i++){
        for(std::unordered_map<int, double>::iterator it = L_val[i].begin(); it != L_val[i].end(); it++){
            L_out<<it->first+1<<" "<<i+1<<" "<<it->second<<std::endl;
        }
    }
    L_out.close();
    U_out.close();

    free(x_pos);
    free(y_pos);
    free(val);

    return 0;
}