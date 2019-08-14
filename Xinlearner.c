
//n入力 隠れ層 1層sigmoid 出力step

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
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

double inner_product1(int a,long pattern){
    double Y = 0;
    for(int i=0 ; i < n ; i++){
        Y += W[n * a + i] * In[pattern * n + i];
    }
    Y += B[a];
    return Y;
}

double inner_product2(){
    double Y = 0;
    for(int i=0 ; i < n ; i++){
        Y += W[n * n + i] * S[i];
    }
    Y += B[n];
    return Y;
}

double step(double a){
    double Y;
    if(a == 0.0){
        Y = 0;
    }else{
        Y = (a / fabs(a) + 1) / 2;
    }
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
    else{
        memset(tmp,0,sizeof(double) * size);
    }
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
    long pattern = pow(2,n);
    long pc = 0;
    printf("pattern : %d\n",pattern);

    In = (double *)pre_matrix(pattern * n);
    printf("In : ok\n");
    T = (double *)pre_matrix(pattern);
    printf("T  : ok\n");
    W = (double *)pre_matrix((n+1) * n);
    printf("W  : ok\n");
    B = (double *)pre_matrix(n+1);
    printf("B  : ok\n");
    S = (double *)pre_matrix(n);
    printf("S  : ok\n");
    test = (double *)pre_matrix(pattern);
    printf("test : ok\n");
    dY = (double *)pre_matrix(pattern);
    printf("dY : ok\n");
    test = (double *)pre_matrix(pattern);
    printf("test : ok\n");
    dW = (double *)pre_matrix((n+1) * n);
    printf("dW : ok\n");
    dB = (double *)pre_matrix(n+1);
    printf("dB : ok\n");
    //入力パターンセッティング
    for(long i=0 ; i < pattern ; i++){
        printf("        pattern[%3d]\n",i);
        for(int j=0 ; j < n ; j++){
            In[i * n + j] = (0b1 & (i >> j));
            printf("In[%3d] : %lf\n",j,In[i * n + j]);
        }
    }
    
    printf("input teatch data\n");
    for(long i=0 ; i < pattern ; i++){
        print01(n,i);
        printf(" -> ");
        scanf("%lf",T + i);
    }

 
    printf("input learn rate : ");
    scanf("%lf",&lr);
    long count;
    printf("input repeat count : ");
    scanf("%d",&count);
    for(long aj=0 ; aj <= count ; aj++){
        //入力・算出
        for(int i=0 ; i < pattern ; i++){
            double Y;
            Y = exe(i);
            printf("Y : %8lf\n",Y);
                //調整
            if((Y - T[i]) == 0);
            else{
                printf("! ajust[%d] !   E : %8lf\n",i,Y - T[i]);
            
                //中間層
                for(int j=0 ; j < n ; j++){
                    for(int k=0 ; k < n ; k++){
                        dW[j * n + k] = (Y - T[i]) * W[n * n + j] * S[j] * (1 - S[j]) * In[i * n + k] * lr;
                        W[j * n + k] -= dW[j * n + k];

                    }
                    dB[j] = (Y - T[i]) * W[n * n + j] * S[j] * (1 - S[j]) * lr;
                    B[j] -= dB[j];
                    
                    printf("%d bias : %8lf(%8lf)\n",j,B[j],(-1 * dB[j]));
                }

                //出力層
                //step
                for(int k=0 ; k < n ; k++){
                    dW[n * n + k] = (Y - T[i]) * S[k] * lr;
                    W[n * n + k] -= dW[n * n + k];
                }
                dB[n] = (Y - T[i]) * lr; 
                B[n] -= dB[n];
                
                printf("output bias : %8lf(%8lf)\n",B[n],(-1 * dB[n]));   
            }
        }

        //テスト
        for(long i=0 ; i < pattern ; i++){
            dY[i] = exe(i) - test[i];
            test[i] = exe(i);
        }

        for(int j=0 ; j < pattern ; j++){
            if(test[j] == T[j]);
            else{
                flug = 1;
            }
        }

        printf("     %7d times     |  ",aj);
        for(long i=0 ; i < pattern ; i++){
            print01(n,i);
            printf(" -> ");
            printf("%lf(%lf)  |  ",test[i],dY[i]);
        }
        printf("\n");
        if(flug == 0) break;
        else{}
        flug = 0;
    }

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
