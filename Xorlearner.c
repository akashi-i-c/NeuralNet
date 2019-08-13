#include<stdio.h>
#include<math.h>

#define patern 4
#define part 2

double In[4][2] = {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}};
double T[4] = {0.0,1.0,1.0,0.0};
double W[3][2] = {{0,1.0},{1.0,0},{0,0}};
double B[] = {0,0,1};
double S[3][2];

double lr;
double test[4];
double dY[4];
double dW[3][2];
double dB[3];

int flug = 0;

double step(double a,double b,int n){
    double Y;
    Y = a*W[n][0] + b*W[n][1] + B[n];
    Y = (Y/fabs(Y) + 1) / 2;
    return Y;
}

double sigmoid(double a,double b,int n){
    double Y;
    Y = a*W[n][0] + b*W[n][1] + B[n];
    Y = 1 / (1 + exp(-1*Y));
    return Y;
}

double exe(int i){
    double Y;
    for(int j=0 ; j < part ; j++){
        S[j][0] = In[i][0];
        S[j][1] = In[i][1];
    }
    S[2][0] = sigmoid(S[0][0],S[0][1],0);
    S[2][1] = sigmoid(S[1][0],S[1][1],1);
    //Y = sigmoid(S[2][0],S[2][1],2);
    Y = step(S[2][0],S[2][1],2);
    return Y;
}

int main(){
    printf("input learn rate : ");
    scanf("%lf",&lr);
    long count;
    printf("input repeat count : ");
    scanf("%d",&count);
    for(long aj=0 ; aj <= count ; aj++){
        //入力・算出
        for(int i=0 ; i < patern ; i++){
            double Y;
            Y = exe(i);
            printf("Y : %8lf\n",Y);
                //調整
            if((Y - T[i]) == 0);
            else{
                printf("! ajust[%d] ! : input = ( %3lf , %3lf ) : %8lf %8lf\n",i,S[0][0],S[0][1],S[2][0],S[2][1]);
                //中間層
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
                //sigmoid - sigmoid
                /*W[0][0] -= (Y - T[i]) * Y * (1 - Y) * W[2][0] * S[2][0] * (1 - S[2][0]) * lr;
                W[0][1] -= (Y - T[i]) * Y * (1 - Y) * W[2][0] * S[2][0] * (1 - S[2][0]) * lr;
                B[0] -= (Y - T[i]) * Y * (1 - Y) * W[2][0] * S[2][0] * (1 - S[2][0]) * lr;
                
                W[1][0] -= (Y - T[i]) * Y * (1 - Y) * W[2][1] * S[2][1] * (1 - S[2][1]) * lr;
                W[1][1] -= (Y - T[i]) * Y * (1 - Y) * W[2][1] * S[2][1] * (1 - S[2][1]) * lr;
                B[1] -= (Y - T[i]) * Y * (1 - Y) * W[2][1] * S[2][1] * (1 - S[2][1]) * lr;
                */
                //step - sigmoid
                /*W[0][0] -= (Y - T[i]) * Y * (1 - Y) * W[2][0] * S[0][0] * lr;
                W[0][1] -= (Y - T[i]) * Y * (1 - Y) * W[2][0] * S[1][0] * lr;
                B[0] -= (Y - T[i]) * Y * (1 - Y) * W[2][0] * lr;
                
                W[1][0] -= (Y - T[i]) * Y * (1 - Y) * W[2][1] * S[0][1] * lr;
                W[1][1] -= (Y - T[i]) * Y * (1 - Y) * W[2][1] * S[1][1] * lr;
                B[1] -= (Y - T[i]) * Y * (1 - Y) * W[2][1] * lr;
                */
                printf("w11 : %8lf(%8lf)  |  w12 : %8lf(%8lf)  |  w1b : %8lf(%8lf)\n",
                        W[0][0],(-1 * dW[0][0]),
                        W[0][1],(-1 * dW[0][1]),
                        B[0],(-1 * dB[0]));
                printf("w21 : %8lf(%8lf)  |  w22 : %8lf(%8lf)  |  w2b : %8lf(%8lf)\n",
                        W[1][0],(-1 * dW[1][0]),
                        W[1][1],(-1 * dW[1][1]),
                        B[1],(-1 * dB[1]));

                //出力層
                //printf("%lf\n",(Y - T[i]));
                //printf("%lf\n",lr);
                //printf("%lf\n",(Y - T[i]) * S[2][0] * lr);
                //step
                dW[2][0] = (Y - T[i]) * S[2][0] * lr;
                dW[2][1] = (Y - T[i]) * S[2][1] * lr;
                dB[2] = (Y - T[i]) * lr; 
                W[2][0] -= dW[2][0];
                W[2][1] -= dW[2][1];
                B[2] -= dB[2];
                //sigmoid
                /*W[2][0] -= (Y - T[i]) * Y * (1 - Y) * S[2][0] * lr;
                W[2][1] -= (Y - T[i]) * Y * (1 - Y) * S[2][1] * lr;
                B[3] -= (Y - T[i]) * Y * (1 - Y) * lr;*/

                printf("w31 : %8lf(%8lf)  |  w32 : %8lf(%8lf)  |  w3b : %8lf(%8lf)\n",
                        W[2][0],(-1 * dW[2][0]),
                        W[2][1],(-1 * dW[2][1]),
                        B[2],(-1 * dB[2]));   
            }
        }

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
        printf("     %7d times     |  %8lf(%8lf)  %8lf(%8lf)  %8lf(%8lf)  %8lf(%8lf)\n",aj,test[0],dY[0],test[1],dY[1],test[2],dY[2],test[3],dY[3]);
        if(flug == 0) break;
        else{}
        flug = 0;
    }

    printf("finish\n");
    return 0;
}
