
//n入力 隠れ層 1層sigmoid 出力step

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double *In;
double *T;
double *W;
double *B;
double *S;

double lr;
double *test;
double *dY;
double *dW;
double *dB;

int n;
int flug = 0;

double inner_product1(int a,long patern){
    double Y = 0;
    for(int i=0 ; i < n ; i++){
        Y += W[n * a + i] * In[patern * n + i];
    }
    Y += B[a];
    return Y;
}

double inner_product2(){
    double Y = 0;
    for(int i=0 ; i < n ; i++){
        Y += W[n * (n - 1) + i] * S[i];
    }
    Y += B[n];
    return Y;
}

double step(double a){
    double Y;
    Y = (a / fabs(a) + 1) / 2;
    return Y;
}

double sigmoid(double a){
    double Y;
    Y = 1 / (1 + exp(-1 * a));
    return Y;
}

double exe(int i){
    double Y;
    for(int j=0 ; j < n ; j++){
        S[j] = sigmoid(inner_product1(j,i));
    }
    Y = inner_product2();
    Y = step(Y);
    return Y;
}

void* pre_matrix(int size){
    void *tmp;
    tmp = malloc(sizeof(double) * size);
    if(tmp == NULL){
        printf("pre_matrix error\n");
        free(In);
        free(T);
        free(W);
        free(B);
        free(S);
        free(test);
        free(dY);
        free(dW);
        free(dB);
        exit(1);
    }
    else{}
    printf("address : %p  ",tmp);
    return tmp;
}

void print01(int bit,long a){
    char *bi;
    bi = (char *)malloc(sizeof(char) * bit);
    if(bi == NULL){
        printf("pre_matrix error\n");
        free(In);
        free(T);
        free(W);
        free(B);
        free(S);
        free(test);
        free(dY);
        free(dW);
        free(dB);
        free(bi);
        exit(1);
    }
    else{}
    for(int i=0 ; i < bit ; i++){
        char tmp;
        tmp = 0b1 & (a >> i);
        if(tmp == 0){
            bi[bit - 1 - i] = '0';
        }else{
            bi[bit - 1 - i] = '1';
        }
    }
    for(int i=0 ; i < bit ; i++){
        printf("%c",bi[i]);
    }
    free(bi);
}

int main(){
    printf("How many input ? -> ");
    scanf("%d",&n);
    long patern = pow(2,n);
    long pc = 0;
    printf("patern : %d\n",patern);

    In = (double *)pre_matrix(patern * n);
    printf("In : ok\n");
    T = (double *)pre_matrix(patern);
    printf("T  : ok\n");
    W = (double *)pre_matrix((n+1) * n);
    printf("W  : ok\n");
    B = (double *)pre_matrix(n+1);
    printf("B  : ok\n");
    S = (double *)pre_matrix(n);
    printf("S  : ok\n");
    test = (double *)pre_matrix(patern);
    printf("test : ok\n");
    dY = (double *)pre_matrix(patern);
    printf("dY : ok\n");
    dW = (double *)pre_matrix((n+1) * n);
    printf("dW : ok\n");
    dB = (double *)pre_matrix(n+1);
    printf("dB : ok\n");
    //入力パターンセッティング
    for(long i=0 ; i < patern ; i++){
        printf("        patern[%3d]\n",i);
        for(int j=0 ; j < n ; j++){
            In[i * n + j] = (0b1 & (i >> j));
            printf("In[%3d] : %lf\n",j,In[i * n + j]);
        }
    }
    
    printf("input teatch data\n");
    for(long i=0 ; i < patern ; i++){
        print01(n,i);
        printf(" -> ");
        scanf("%lf",T + i);
        //printf("%lf\n",T[i]);
    }

/*    
    printf("input learn rate : ");
    scanf("%lf",&lr);
    long count;
    printf("input repeat count : ");
    scanf("%d",&count);
    for(long aj=0 ; aj <= count ; aj++){
        *///入力・算出
        for(int i=0 ; i < patern ; i++){
            double Y;
            Y = exe(i);
            printf("Y : %8lf\n",Y);
                //調整
            if((Y - T[i]) == 0);
            else{
                printf("! ajust[%d] !   E : %8lf\n",i,Y - T[i]);
            
            /*//中間層
                //sigmoid-step
                dW[0][0] = (Y - T[i]) * W[2][0] * S[2][0] * (1 - S[2][0]) * S[0][0] * lr;
                dW[0][1] = (Y - T[i]) * W[2][0] * S[2][0] * (1 - S[2][0]) * S[0][1] * lr;
                dB[0] = (Y - T[i]) * W[2][0] * S[2][0] * (1 - S[2][0]) * lr;
                W[0][0] -= dW[0][0];
                W[0][1] -= dW[0][1];
                B[0] -= dB[0];

                dW[1][0] = (Y - T[i]) * W[2][1] * S[2][1] * (1 - S[2][1]) * S[1][0] * lr;
                dW[1][1] = (Y - T[i]) * W[2][1] * S[2][1] * (1 - S[2][1]) * S[1][1] * lr;
                dB[1] = (Y - T[i]) * W[2][1] * S[2][1] * (1 - S[2][1]) * lr;    
                W[1][0] -= dW[1][0];
                W[1][1] -= dW[1][1];
                B[1] -= dB[1];
                
                printf("w11 : %8lf(%8lf)  |  w12 : %8lf(%8lf)  |  w1b : %8lf(%8lf)\n",
                        W[0][0],(-1 * dW[0][0]),
                        W[0][1],(-1 * dW[0][1]),
                        B[0],(-1 * dB[0]));
                printf("w21 : %8lf(%8lf)  |  w22 : %8lf(%8lf)  |  w2b : %8lf(%8lf)\n",
                        W[1][0],(-1 * dW[1][0]),
                        W[1][1],(-1 * dW[1][1]),
                        B[1],(-1 * dB[1]));

                //出力層
                //step
                dW[2][0] = (Y - T[i]) * S[2][0] * lr;
                dW[2][1] = (Y - T[i]) * S[2][1] * lr;
                dB[2] = (Y - T[i]) * lr; 
                W[2][0] -= dW[2][0];
                W[2][1] -= dW[2][1];
                B[2] -= dB[2];
                
                printf("w31 : %8lf(%8lf)  |  w32 : %8lf(%8lf)  |  w3b : %8lf(%8lf)\n",
                        W[2][0],(-1 * dW[2][0]),
                        W[2][1],(-1 * dW[2][1]),
                        B[2],(-1 * dB[2]));   
            */
            }
        }
/*
        dY[0] = exe(0) - test[0];
        dY[1] = exe(1) - test[1];
        dY[2] = exe(2) - test[2];
        dY[3] = exe(3) - test[3];

        //テスト
        test[0] = exe(0);
        test[1] = exe(1);
        test[2] = exe(2);
        test[3] = exe(3);

        for(int j=0 ; j < patern ; j++){
            if(test[j] == T[j]);
            else{
                flug = 1;
            }
        }
        printf("     %7d times     |  (0,0) -> %8lf(%8lf) (1,0) -> %8lf(%8lf)  (0,1) -> %8lf(%8lf)  (1,1) -> %8lf(%8lf)\n",aj,test[0],dY[0],test[1],dY[1],test[2],dY[2],test[3],dY[3]);
        if(flug == 0) break;
        else{}
        flug = 0;
    }
*/
    printf("finish\n");
    free(In);
    free(T);
    free(W);
    free(B);
    free(S);
    free(test);
    free(dY);
    free(dW);
    free(dB);
    return 0;
}
