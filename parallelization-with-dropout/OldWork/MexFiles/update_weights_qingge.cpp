#include <iostream>
#include <vector>
#include <assert.h>
#include <math.h>
#include <iomanip>
using namespace std;

int main() {

double input[4]={0,0,0,0};

double input1[4]={tanh(0)/0.5, tanh(0)/0.5, tanh(0)/0.5,tanh(0)/0.5};

cout<<"Printing input layer Data:"<<endl;
    for(int i = 0; i < 4; i++){
        cout << input1[i] << endl;
    }
  
vector<double> output_vals1;
  
double weights1[6][5] = {{0.10511860249074981,    -0.8691958077685074,    3.9107265744512154,    3.8296505582151905,    0.0431373966662373}, {0.8052531277712192,    0.11608271973910032,    5.081202415079452,    -1.666910901747036,    1.2461853285676618},{0.14211713490656408,    0.2723755030710997,    4.040974596086221,    2.0482239539091487,    0.14506617911530964},{-0.3952723230078815,    -1.4229869215305775,    4.2551314235012185,    5.561432608781822,    2.512020072447385E-4},{-0.2772247462559281,    0.6999358816350526,    1.6369899052747476,    2.6788819705306155,    -0.10953612423359192},{
0.37612754523479003,    1.2857167349312448,    -2.973060687194107,    8.095548964772654,    0.028991336931277124}};


  //perform matrix multiplication
    for(int out = 0; out < 6; out++){
        double sum1 = 0.0;
        for(int w = 0; w < 4; w++){
            sum1 += weights1[out][w] * input1[w];
        }
        sum1 += weights1[out][4]; //account for the bias
        
        output_vals1.push_back(tanh(sum1));
    }

   cout<<"Printing first hidden layer Data:"<<endl;
    for(int i = 0; i < 6; i++){
        cout << output_vals1[i] << endl;
    }

//first hidden layer input=output_vals1
vector<double> output_vals2;

double weights2[6][7]={{1.440539114493213,    -0.27253071839005777,    0.5278868902219289,    1.3712226806164334,    2.25513928618451,    1.3948446255239006,    0.3449374254994523
},{3.3789817246548357,    0.19860814862310894,    3.4592707210714577,    -1.9110270294298068,    0.2247519084049887,    -1.0022103471763135,    -0.4753830547340122
},{-0.3590224302196529,    1.2176551644649063,    3.145578151429633,    1.8631207326452712,    3.708974043074285,    -0.09608244193951347,    1.5661350150973756
},{4.031099888420788,    -2.6851875859289094,    -2.7498688647349647,    2.7486598886675715,    2.439552173754654,    5.660170953027147,    0.925728746264457
},{-0.6221529426089921,    0.7320648747641353,    4.212370496471141,    -4.0812165587839555,    1.5473829764451348,    -6.45653407631204,    -0.817547511050558
},{1.5225000509529556,    -1.0360350040097754,    1.7030720130819914,    0.5344327232788693,    0.6307629342167956,    1.0930386330505284,    0.07424903099042285
}};


    //perform matrix multiplication
    for(int out = 0; out < 6; out++){
        double sum2 = 0.0;
        for(int w = 0; w < 6; w++){
            sum2 += weights2[out][w] * output_vals1[w];
        }
        sum2 += weights2[out][6]; //account for the bias
       
        output_vals2.push_back(tanh(sum2));
    }

cout<<"Printing second hidden layer output Data:"<<endl;
    for(int i = 0; i < 6; i++){
        cout << output_vals2[i] << endl;
    }
//second hidden layer input=output_vals2
vector<double> output_vals3;
  
double weights3[2][7]={{-1.7111553944356483,    -0.44719687703118854,    -2.6145089128562864,    -5.956955188009836,    0.9585898449575091,    1.641209174573893,    -2.4405474217252867
},{1.3179776848229032,    1.0386077175091335,    2.191677954355899,    -0.5152838015317462,    0.9730610147224401,    3.0380696863621965,    -1.3866542032979876
}};

   //perform matrix multiplication
    for(int out = 0; out < 2; out++){
        double sum3 = 0.0;
        for(int w = 0; w < 6; w++){
            sum3 += weights3[out][w] * output_vals2[w];
        }
        sum3 += weights3[out][6]; //account for the bias
   
        output_vals3.push_back(tanh(sum3));
    }

double output1=output_vals3[0];
double output2=output_vals3[1]; 

cout<<"Printing output layer Data:"<<endl;
    for(int i = 0; i <2; i++){
      
        cout << output_vals3[i] << endl;
    }   


  
///////////////////////////////////////////////////////////
/////the derivative of output for inputs
//////////////////////////////////////////////////////////
vector<double> output_vals11;
vector<double> output_vals12;
vector<double> output_vals13;
vector<double> output_vals14;

//perform matrix multiplication
  //for(int w = 0; w < 4; w++){
    double sum11 = 0.0;
    double sum12 = 0.0;
    double sum13 = 0.0;
    double sum14 = 0.0;

  
    for(int out = 0; out < 6; out++){
           sum11 = (1 - output_vals1[out]* output_vals1[out]) * ( weights1[out][0] * (1-input1[0]*input1[0]) );
           sum12 = (1 - output_vals1[out]* output_vals1[out]) * ( weights1[out][1] * (1-input1[1]*input1[1]) );
           sum13 = (1 - output_vals1[out]* output_vals1[out]) * ( weights1[out][2] * (1-input1[2]*input1[2]) );
           sum14 = (1 - output_vals1[out]* output_vals1[out]) * ( weights1[out][3] * (1-input1[3]*input1[3]) );
        
        //sum11 += weights1[out][4]; //account for the bias
        
        output_vals11.push_back(sum11);
        output_vals12.push_back(sum12);
        output_vals13.push_back(sum13);
        output_vals14.push_back(sum14);}
    //}
  
vector<double> output_vals21;
vector<double> output_vals22;
vector<double> output_vals23;
vector<double> output_vals24;
 //perform matrix multiplication
    for(int out = 0; out < 6; out++){
        double sum21 = 0.0;
        double sum22 = 0.0;
        double sum23 = 0.0;
        double sum24 = 0.0;
        for(int w = 0; w < 6; w++){
            sum21 += weights2[out][w] * output_vals11[w];
            sum22 += weights2[out][w] * output_vals12[w];
            sum23 += weights2[out][w] * output_vals13[w];
            sum24 += weights2[out][w] * output_vals14[w];
        }
        //sum22 += weights2[out][6]; //account for the bias
       sum21=sum21*(1-output_vals2[out] * output_vals2[out] );
      
       sum22=sum22*(1-output_vals2[out] * output_vals2[out] );
       sum23=sum23*(1-output_vals2[out] * output_vals2[out] );
       sum24=sum24*(1-output_vals2[out] * output_vals2[out] );
      
        output_vals21.push_back(sum21);
        output_vals22.push_back(sum22);
        output_vals23.push_back(sum23);
        output_vals24.push_back(sum24);
    }

  double sum31=0.0;
  double sum32=0.0;
  double sum33=0.0;
  double sum34=0.0;
  
  for(int w = 0; w < 6; w++){
     sum31 += weights3[0][w] * output_vals21[w];
     sum32 += weights3[0][w] * output_vals22[w];
     sum33 += weights3[0][w] * output_vals23[w];
     sum34 += weights3[0][w] * output_vals24[w];
  }

  double sum41=0.0;
  double sum42=0.0;
  double sum43=0.0;
  double sum44=0.0;
  
  for(int w = 0; w < 6; w++){
     sum41 += weights3[1][w] * output_vals21[w];
     sum42 += weights3[1][w] * output_vals22[w];
     sum43 += weights3[1][w] * output_vals23[w];
     sum44 += weights3[1][w] * output_vals24[w];
  }

  double y1x1=2*(1-output1 * output1)*sum31;
  double y1x2=2*(1-output1 * output1)*sum32;
  double y1x3=2*(1-output1 * output1)*sum33;
  double y1x4=2*(1-output1 * output1)*sum34;

  double y2x1=2*(1-output2 * output2)*sum41;
  double y2x2=2*(1-output2 * output2)*sum42;
  double y2x3=2*(1-output2 * output2)*sum43;
  double y2x4=2*(1-output2 * output2)*sum44;
  
  cout<<"Printing y1 weights dirivative for inputs:"<<endl; 
  cout<<setprecision(15)<<y1x1 <<' '<< y1x2<<' '<<y1x3<<' ' <<y1x4<<endl;

  cout<<"Printing y2 weights dirivative for inputs:"<<endl; 
  cout<<setprecision(15)<<y2x1 <<' '<< y2x2<<' '<<y2x3<<' ' <<y2x4<<endl;

  cout<<endl;

  double Dnet_Didq[2][2]={{y1x1,y1x2},{y2x1,y2x2}};
  double Dnet_Dhist_err[2][2]={{y1x3,y1x4},{y2x3,y2x4}};

///////////////////////////////////////////////////////////
  ///////////////derivative for weights3
  /////////////////////////////////////////////////////////

  double weights_derivative[2][86];

  double weights34_out1[6];
  for(int out = 0; out < 6; out++){
     weights34_out1[out]=(1-output1 * output1)*output_vals2[out];
    weights_derivative[0][out]=weights34_out1[out];
    weights_derivative[1][out]=0;
    }

 
  
  cout<<endl;
  cout<<"Printing y1 derivative for weights3"<<endl;
  for(int out = 0; out < 6; out++){
     cout<<weights34_out1[out]<<" ";
    }

    double weights34_out2[6];
  for(int out = 0; out < 6; out++){
     weights34_out2[out]=(1-output2 * output2)*output_vals2[out];
     weights_derivative[0][out+7]=0;
     weights_derivative[1][out+7]=weights34_out2[out];
    }////////continue tonight
  cout<<endl;
  cout<<"Printing y2 derivative for weights3"<<endl;
  for(int out = 0; out < 6; out++){
     cout<<weights34_out2[out]<<" ";
    }
  
///////////////////////////////////////////////////////////
  ///////////////derivative for weights2
  /////////////////////////////////////////////////////////


  double weights23_out1[6][6];
  for(int out = 0; out < 6; out++){
        for(int input = 0; input < 6; input++){
            weights23_out1[out][input] = (1-output1 * output1)* weights3[0][out] * output_vals1[input]*(1-output_vals2[out] * output_vals2[out]);

          weights_derivative[0][input+14+out*7]=weights23_out1[out][input];
        }
  }




  
  cout<<endl;
  cout<<"Printing out1 for weights23_out1 derivative matrix"<<endl;
   for(int out = 0; out < 6; out++){
        for(int input = 0; input < 6; input++){
          cout<<weights23_out1[out][input]<<" ";
    
          }
     }
  

  double weights23_out2[6][6];
  for(int out = 0; out < 6; out++){
        for(int input = 0; input < 6; input++){
            weights23_out2[out][input] = (1-output2 * output2)* weights3[1][out] * output_vals1[input]*(1-output_vals2[out] * output_vals2[out]);

            weights_derivative[1][input+14+out*7]=weights23_out2[out][input];
        }
  }


  cout<<endl;
  cout<<"Printing out2 for weights23_out2 derivative matrix"<<endl;
   for(int out = 0; out < 6; out++){
        for(int input = 0; input < 6; input++){
          cout<<weights23_out2[out][input]<<" ";
    
          }
     }

//////////////////////////////////////////////////
///////////// The dirivative for weights1
//////////////////////////////////////////////////
double weights12_out1[6][5];


  double output_vals111[6];

  for(int out=0; out<6; out++){
    double sumOf = 0;
     for(int input = 0; input < 6; input++){
        sumOf += weights3[0][input] * (1- 
        output_vals2[input] * output_vals2[input]) * 
        weights2[input][out] * (1-output_vals1[out] * 
        output_vals1[out] );
    //weights12_out1[input][out]=(1-output1 * output1)*sumOf;
        
     }
     output_vals111[out]=sumOf;
  }

for(int input=0; input<6; input++){
  for(int out=0; out<4; out++){
    weights12_out1[input][out]=(1-output1 * output1)*input1[out]*output_vals111[input];
}
  }
  
for(int out=0; out<6; out++){
   weights12_out1[out][4]=(1-output1 * output1)*output_vals111[out];
  }
  
cout<<endl;
cout<<"Printing out1 for derivative for weights1"<<endl;

  
for(int input=0; input<6; input++){
  for(int out=0; out<5; out++){
    cout<<weights12_out1[input][out]<<" ";
       weights_derivative[0][out+56+input*5]=weights12_out1[input][out];
    }
}
///////////////////////////////////////////////////////////
  //////////////derivative for weights1 of out2
  ////////////////////////////////////////////
double weights12_out2[6][5];


  double output_vals112[6];

  for(int out=0; out<6; out++){
    double sumOf = 0;
     for(int input = 0; input < 6; input++){
        sumOf += weights3[1][input] * (1- 
        output_vals2[input] * output_vals2[input]) * 
        weights2[input][out] * (1-output_vals1[out] * 
        output_vals1[out] );
    //weights12_out1[input][out]=(1-output1 * output1)*sumOf;
        output_vals112[out]=sumOf;
     }
    
  }

for(int input=0; input<6; input++){
  for(int out=0; out<4; out++){
    weights12_out2[input][out]=(1-output2 * output2)*input1[out]*output_vals112[input];
}
  }

for(int out=0; out<6; out++){
   weights12_out2[out][4]=(1-output2 * output2)*output_vals112[out];
  }
  
cout<<endl;
cout<<"Printing out2 for derivative for weights1"<<endl;

  
for(int input=0; input<6; input++){
  for(int out=0; out<5; out++){
    cout<<weights12_out2[input][out]<<" ";
        weights_derivative[1][out+56+input*5]=weights12_out2[input][out];
    }
}

cout<<endl;
cout<<"Printing derivative for output layer bias";
cout<<endl;
cout<<(1-output1 * output1);
cout<<endl;
cout<<(1-output2 * output2);

  weights_derivative[0][6]=(1-output1 * output1);
  weights_derivative[1][6]=0;
  weights_derivative[0][13]=0;
  weights_derivative[1][13]=(1-output2 * output2);

cout<<endl;
cout<<"Printing out1 derivative for second hidden layer bias";
cout<<endl;

for(int i=0; i<6; i++){
  cout<<(1-output1 * output1)*weights3[0][i]*(1- 
        output_vals2[i] * output_vals2[i])<<" ";
  
      weights_derivative[0][20+i*7]=(1-output1 * output1)*weights3[0][i]*(1- 
        output_vals2[i] * output_vals2[i]);
  }


cout<<endl;
cout<<"Printing out2 derivative for second hidden layer bias";
cout<<endl;

for(int i=0; i<6; i++){
  cout<<(1-output2 * output2)*weights3[1][i]*(1- 
        output_vals2[i] * output_vals2[i])<<" ";

   weights_derivative[1][20+i*7]=(1-output2 * output2)*weights3[1][i]*(1- 
        output_vals2[i] * output_vals2[i]);
  }
//cout<<(1-output1 * output1)*weights3[0][0]*(1- 
        //output_vals2[0] * output_vals2[0]);

cout<<endl;
cout<<endl;
cout<<endl;
cout<<"Printing 2*86 weights matrix"<<endl;
for(int i=0; i<2; i++){
  for(int j=0; j<86; j++){
    cout<<weights_derivative[i][j]<<" ";
  }
}

cout<<endl;
cout<<endl;
cout<<endl;
cout<<"Printing Dnet_Didq matrix"<<endl;
for(int i=0; i<2; i++){
  for(int j=0; j<2; j++){
    cout<<Dnet_Didq[i][j]<<" ";
  }
}


cout<<endl;
cout<<"Printing Dnet_Dhist_err matrix"<<endl;
for(int i=0; i<2; i++){
  for(int j=0; j<2; j++){
    cout<<Dnet_Dhist_err[i][j]<<" ";
  }
}



  
}

