/////////////////////////////////
// Mayank Roy
// Assignment : Dr.Sebastian Scherer
// Section : 2
///////////////////////

#include "ros/ros.h"
#include<stdio.h>
#include "std_msgs/String.h"
#include<iostream>
#include <sstream>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using namespace Eigen;
using namespace std;

Matrix3f MyRotation(const Vector3f YPR)
{
    Matrix3f matYaw(3, 3), matRoll(3, 3), matPitch(3, 3), matRotation(3, 3);
    float yaw = YPR[2]*M_PI / 180;
    float pitch = YPR[0]*M_PI / 180;
    float roll = YPR[1]*M_PI / 180;

    matYaw << cos(yaw), -sin(yaw), 0.0f,            //rotation around Z
        sin(yaw), cos(yaw), 0.0f,  
        0.0f, 0.0f, 1.0f;

    matPitch << cos(pitch), 0.0f, sin(pitch),       //rotation around Y
        0.0f, 1.0f, 0.0f,   
        -sin(pitch), 0.0f, cos(pitch);

    matRoll << 1.0f, 0.0f, 0.0f,                    //rotation around X
        0.0f, cos(roll), -sin(roll),   
        0.0f, sin(roll), cos(roll);

    matRotation = matRoll*matPitch*matYaw;         //Intrinsic rotation with Left Multiplication and Rotation Order - Roll,Pitch,Yaw

    Quaternionf quatFromRot(matRotation);

    quatFromRot.normalize(); 

    return matRotation; 
}


int main(int argc, char **argv)

{

  ros::init(argc, argv, "rotate");
  ros::NodeHandle n;
  std_msgs::String msg;
  stringstream ss;
  ss << "hello \n " ;
  msg.data = ss.str();
  float yaw, pitch, roll;


   ROS_INFO("%s", msg.data.c_str());
   printf("Please Enter Yaw : ");
   scanf("%f",&yaw);	
   printf("Please Enter Pitch : ");
   scanf("%f",&pitch);	
   printf("Please Enter Roll : ");
   scanf("%f",&roll);
   Vector3f YPR(pitch,roll,yaw);

   cout<<MyRotation(YPR)<<endl;
   getchar();
	
   yaw = YPR[2]*M_PI / 180;
   pitch = YPR[0]*M_PI / 180;
   roll = YPR[1]*M_PI / 180;

   AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
   AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
   AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());

   Quaternion<double> q = rollAngle * pitchAngle * yawAngle ; //Intrinsic rotation with Left Multiplication and 
                                                                     //Rotation Order - Roll,Pitch,Yaw

   Matrix3d rotationMatrix = q.matrix();       //Cross Verification

//   cout<<rotationMatrix<<endl;    //Done finding Matrix notation.

   cout<<"Quat.w = "<<q.w()<<"  Quat.x = "<<q.x()<<"  Quat.y = "<<q.y()<<"  Quat.z = "<<q.z()<<endl<<endl;
   cout<<"End of Rotation Calculations"<<endl<<endl;
   
   Quaternion<double> Qa,Qb,Qc,Qd,Qe,Qf; 
   Qa.w() = 0.800103; Qa.x() = 0.191342; Qa.y() = 0.331414; Qa.z() = 0.46194;    
   Qb.w() = 0.866025; Qb.x() = -0.5; Qb.y() = 0; Qb.z() = 0;                     

   Qc = Qa*Qb;
   Qd = Qb*Qa;

  
   cout<<"Qc.w = "<<Qc.w()<<"  Qc.x = "<<Qc.x()<<"  Qc.y = "<<Qc.y()<<"  Qc.z = "<<Qc.z()<<endl;
   cout<<"Qd.w = "<<Qd.w()<<"  Qd.x = "<<Qd.x()<<"  Qd.y = "<<Qd.y()<<"  Qd.z = "<<Qd.z()<<endl;

   Qe = Qa * Qb.inverse();
   Qf = Qe * Qb;

   cout<<"Qa.w = "<<Qa.w()<<"  Qa.x = "<<Qa.x()<<"  Qa.y = "<<Qa.y()<<"  Qa.z = "<<Qa.z()<<endl;
   cout<<"Qf.w = "<<Qf.w()<<"  Qf.x = "<<Qf.x()<<"  Qf.y = "<<Qf.y()<<"  Qf.z = "<<Qf.z()<<endl;
	

   getchar();
   return 0;
}     
