#include <iostream>
#include <vector>
#include <assert.h>
//#include "Layer.h"
//#include "Model.h"
#include <cmath>
using namespace std;

int main()
{
//initialize variables
double Ts, b, a, n;
b=1;
a=0;
n=1000;
Ts = (b-a)/n;
int tt=0;
double  kp = 30.618621784789724;
double idq[2][1]={0,0};
//double idq2[2][1]={0,0};
double idq_ref[2][1]={1,0};
double edq[2][1];
double vdq[2][1] = {14,-2};
double SumOfedq[2][1]={0,0};
double sdq[2][1], udq[2][1] ;
//long double R[2][1] = {1,1};
double R[2][1] ;
double A[2][2] = {{ 0.920525055277549,  0.364461652184452}, {-0.364461652184452, 0.920525055277549}};
double B[2][2] ={{ -0.038866915258523, -0.007401576588668}, {0.007401576588668, -0.038866915258523}};

   

vector<vector <double> > train_data;
train_data.push_back({0, 0, 0, 0, 1});
double input[4];
double output1[6];
double output2[6];
double output3[2];

double w1[6][5] = {{0.10511860249074981,    -0.8691958077685074,    3.9107265744512154,    3.8296505582151905,    0.0431373966662373}, {0.8052531277712192,    0.11608271973910032,    5.081202415079452,    -1.666910901747036,    1.2461853285676618},{0.14211713490656408,    0.2723755030710997,    4.040974596086221,    2.0482239539091487,    0.14506617911530964},{-0.3952723230078815,    -1.4229869215305775,    4.2551314235012185,    5.561432608781822,    2.512020072447385E-4},{-0.2772247462559281,    0.6999358816350526,    1.6369899052747476,    2.6788819705306155,    -0.10953612423359192},{
0.37612754523479003,    1.2857167349312448,    -2.973060687194107,    8.095548964772654,    0.028991336931277124}};

double w2[6][7]={{1.440539114493213,    -0.27253071839005777,    0.5278868902219289,    1.3712226806164334,    2.25513928618451,    1.3948446255239006,    0.3449374254994523
},{3.3789817246548357,    0.19860814862310894,    3.4592707210714577,    -1.9110270294298068,    0.2247519084049887,    -1.0022103471763135,    -0.4753830547340122
},{-0.3590224302196529,    1.2176551644649063,    3.145578151429633,    1.8631207326452712,    3.708974043074285,    -0.09608244193951347,    1.5661350150973756
},{4.031099888420788,    -2.6851875859289094,    -2.7498688647349647,    2.7486598886675715,    2.439552173754654,    5.660170953027147,    0.925728746264457
},{-0.6221529426089921,    0.7320648747641353,    4.212370496471141,    -4.0812165587839555,    1.5473829764451348,    -6.45653407631204,    -0.817547511050558
},{1.5225000509529556,    -1.0360350040097754,    1.7030720130819914,    0.5344327232788693,    0.6307629342167956,    1.0930386330505284,    0.07424903099042285
}};

double w3[2][7]={{-1.7111553944356483,    -0.44719687703118854,    -2.6145089128562864,    -5.956955188009836,    0.9585898449575091,    1.641209174573893,    -2.4405474217252867
},{1.3179776848229032,    1.0386077175091335,    2.191677954355899,    -0.5152838015317462,    0.9730610147224401,    3.0380696863621965,    -1.3866542032979876
}};



while (tt<50){
//calculate edq
for(int i=0; i<2; i++)
{
    edq[i][0] = -(idq_ref[i][0] - idq[i][0]);// idq- idq_ref
    SumOfedq[i][0]=SumOfedq[i][0]+edq[i][0];
    sdq[i][0] = Ts * (SumOfedq[i][0] - edq[i][0]/2);
}
    cout<<"printing sdq at iteration"<<" "<<tt<<endl;
    for(int i=0;i<2;i++)
    {
        
       cout<<sdq[i][0]<<" ";
        cout<<"\n";
    }
   
 //set inputs for NN
    int ind = rand() % (train_data.size());
    train_data[ind][0] = tanh(edq[0][0]/0.5);
    train_data[ind][1] = tanh(edq[1][0]/0.5);
    train_data[ind][2] = tanh(sdq[0][0]/0.5);
    train_data[ind][3] = tanh(sdq[1][0]/0.5);

   

    //Input Layer
    //vector<double> output1 = model1.feedForward({train_data[ind][0], train_data[ind][1], train_data[ind][2], train_data[ind][3] });

    
    for(int i=0;i<4;i++){
       input[i]=train_data[ind][i];
}
    
    cout<<"printing train_data"<<endl;
    for(int i=0;i<4;i++)
    {
        
       cout<<input[i]<<" ";
        cout<<"\n";
    }
 


//perform matrix multiplication
    for(int out = 0; out < 6; out++){
        double sum = 0.0;
        for(int w = 0; w < 4; w++){
            sum += w1[out][w] * input[w];
        }
        sum += -w1[out][4]; //account for the bias, the operator in front of w3 should be "-1"
        output1[out]=tanh(sum);
    }

    cout<<"printing output1:"<<endl;
    for(int j = 0; j < 6; j++){
        cout<<output1[j]<<" "<<endl;}
    
    cout<<"printing next item"<<endl;
    //Second Layer
    //vector<double> output2 = model2.feedForward({output1[0], output1[1], output1[2], output1[3], output1[4], output1[5] });


    
//perform matrix multiplication
    for(int out = 0; out < 6; out++){
        double sum = 0.0;
        for(int w = 0; w < 6; w++){
            sum += w2[out][w] * output1[w];
        }
        sum += -w2[out][6]; //account for the bias, the operator in front of w3 should be "-1"
        output2[out]=tanh(sum);
    }

    cout<<"printing output2:"<<endl;
    for(int k = 0; k < 6; k++){
        cout<<output2[k]<<" "<<endl;}
    
    cout<<"printing next item"<<endl;

    //Third Layer
    //vector<double> output = model3.feedForward({output2[0], output2[1], output2[2], output2[3], output2[4], output2[5] });



  
//perform matrix multiplication
    for(int out = 0; out < 2; out++){
        double sum = 0.0;
        for(int w = 0; w <6; w++){
            sum += w3[out][w] * output2[w];
        }
        sum += -w3[out][6]; //account for the bias, the operator in front of w3 should be "-1"
        output3[out]=tanh(sum);
    }
    
    cout<<"printing output3:"<<endl;
    for(int m = 0; m < 2; m++){
        cout<<output3[m]<<" "<<endl;}
    


    for(int n = 0; n < 2; n++){
        R[n][0]= output3[n];
    }

    cout<<"printing R:"<<endl;
    for(int t = 0; t < 2; t++){
        cout<<R[t][0]<<" "<<endl;}
    

//calculate udq
for(int j=0; j<2; j++)
{
   udq[j][0] = kp * R[j][0] - vdq[j][0];
}
   
    cout<<"printing udq:"<<endl;
    for(int t = 0; t < 2; t++){
        cout<<udq[t][0]<<" "<<endl;}
    
  
    
    
        
// calculate idq; the last two digits are different
    for(int i = 0; i < 2; i++){
            double sum = 0.0;
            for(int k = 0; k < 2; k++)
            {
                sum+= A[i][k] * idq[k][0] + B[i][k] * udq[k][0];
               
            }
        idq[i][0]=sum;
    }
    
    cout<<"printing idq:"<<endl;
    for(int t = 0; t < 2; t++){
        cout<<idq[t][0]<<" "<<endl;}



tt=tt+1;

}
    /*
    for(int i=0;i<2;i++)
    {
         cout.precision(15); //the last two digits are different
       cout<<output3[i]<<"";
        cout<<"\n";
    }*/



return 0;
}    