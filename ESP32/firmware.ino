#include <micro_ros_arduino.h>
#include <RMCS2303drive.h>
#include <geometry_msgs/msg/twist.h>
#include <std_msgs/msg/int32.h>
#include "MapFloat.h"
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

RMCS2303 rmcs;  // creation of motor driver object
// slave ids to be set on the motor driver refer to the manual in the reference section
byte slave_id1 = 3;
byte slave_id2 = 1;
byte slave_id3 = 2;
byte slave_id4 = 4;

// Micro-ROS variables
rcl_publisher_t left_ticks_pub;
rcl_publisher_t right_ticks_pub;
rcl_subscription_t sub;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;

geometry_msgs__msg__Twist msg;  // msg variable of data type twist
std_msgs__msg__Int32 lwheel;    // for storing left encoder value
std_msgs__msg__Int32 rwheel;    // for storing right encoder value

// Make sure to specify the correct values here
//*******************************************
double wheel_rad = 0.0500, wheel_sep = 0.260;  // wheel radius and wheel separation in meters.
//******************************************
double w_r = 0, w_l = 0;
double speed_ang;
double speed_lin;
double leftPWM;
double rightPWM;

bool new_command_received = false; // Flag to track new command reception

void messageCb(const void *msg_in)  // cmd_vel callback function definition
{
  const geometry_msgs__msg__Twist *msg = (const geometry_msgs__msg__Twist *)msg_in;
  new_command_received = true;
    speed_lin = fmax(fmin(msg->linear.x, 1.0f), -1.0f);   // limits the linear x value from -1 to 1
    speed_ang = fmax(fmin(msg->angular.z, 1.0f), -1.0f);  // limits the angular z value from -1 to 1
  
    // Kinematic equation for finding the left and right velocities
    w_r = (speed_lin / wheel_rad) + ((speed_ang * wheel_sep) / (2.0 * wheel_rad));
    w_l = (speed_lin / wheel_rad) - ((speed_ang * wheel_sep) / (2.0 * wheel_rad));
  
    if (w_r == 0)
    {
      rightPWM = 0;
      rmcs.Disable_Digital_Mode(slave_id1,0);
      rmcs.Disable_Digital_Mode(slave_id2,0);  // if right motor velocity is zero set right pwm to zero and disabling motors
      rmcs.Disable_Digital_Mode(slave_id3,0);
      rmcs.Disable_Digital_Mode(slave_id4,0);
    }
    else
      rightPWM = mapFloat(fabs(w_r), 0.0, 18.0, 1500,17200);  // mapping the right wheel velocity with respect to Motor PWM values
  
    if (w_l == 0)
    {
      leftPWM = 0;
      rmcs.Disable_Digital_Mode(slave_id1,0);
      rmcs.Disable_Digital_Mode(slave_id2,0);  // if left motor velocity is zero set left pwm to zero and disabling motors
      rmcs.Disable_Digital_Mode(slave_id3,0);
      rmcs.Disable_Digital_Mode(slave_id4,0);
    }
    else
      leftPWM = mapFloat(fabs(w_l), 0.0, 18.0, 1500,17200);  // mapping the right wheel velocity with respect to Motor PWM values
  
    rmcs.Speed(slave_id1,rightPWM);
    rmcs.Speed(slave_id2,rightPWM);
    rmcs.Speed(slave_id3,leftPWM);
    rmcs.Speed(slave_id4,leftPWM);
  
    if (w_r > 0 && w_l > 0)
    {
      rmcs.Enable_Digital_Mode(slave_id1,1);
      rmcs.Enable_Digital_Mode(slave_id2,1);  // forward condition
      rmcs.Enable_Digital_Mode(slave_id3,0);
      rmcs.Enable_Digital_Mode(slave_id4,0);
    }
    else if (w_r < 0 && w_l < 0)
    {
      rmcs.Enable_Digital_Mode(slave_id1,0);
      rmcs.Enable_Digital_Mode(slave_id2,0);  // backward condition
      rmcs.Enable_Digital_Mode(slave_id3,1);
      rmcs.Enable_Digital_Mode(slave_id4,1);
    }
    else if (w_r > 0 && w_l < 0)
    {
      rmcs.Enable_Digital_Mode(slave_id1,1);
      rmcs.Enable_Digital_Mode(slave_id2,1);  // Leftward condition
      rmcs.Enable_Digital_Mode(slave_id3,1);
      rmcs.Enable_Digital_Mode(slave_id4,1);
    }
    else if (w_r < 0 && w_l > 0)
    {
      rmcs.Enable_Digital_Mode(slave_id1,0);
      rmcs.Enable_Digital_Mode(slave_id2,0);  // rightward condition
      rmcs.Enable_Digital_Mode(slave_id3,0);
      rmcs.Enable_Digital_Mode(slave_id4,0);
    }
    else
    {
      rmcs.Brake_Motor(slave_id1,0);
      rmcs.Brake_Motor(slave_id2,0);
      rmcs.Brake_Motor(slave_id3,0);
      rmcs.Brake_Motor(slave_id4,0);  // if none of the above break the motors both in clockwise n anti-clockwise direction
      rmcs.Brake_Motor(slave_id1,1);
      rmcs.Brake_Motor(slave_id2,1);
      rmcs.Brake_Motor(slave_id3,1);
      rmcs.Brake_Motor(slave_id4,1);
    }
    
}

void setup()
{
  // Initialize serial for debugging
  Serial.begin(115200);
  rmcs.Serial_selection(0);  // 0 -> for Hardware serial tx1 rx1 of Arduino Mega
  rmcs.initSerial(9600);
  rmcs.begin(&Serial2, 9600, 16, 17);

  // Micro-ROS initialization
  set_microros_transports();
  allocator = rcl_get_default_allocator();
  rclc_support_init(&support, 0, NULL, &allocator);
  rclc_node_init_default(&node, "esp32_node", "", &support);

  // Create publishers and subscribers
  rclc_publisher_init_default(
      &left_ticks_pub,
      &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
      "left_ticks");
      
  rclc_publisher_init_default(
      &right_ticks_pub,
      &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
      "right_ticks");

  rclc_subscription_init_default(
      &sub,
      &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
      "cmd_vel");

  // Create executor
  rclc_executor_init(&executor, &support.context, 1, &allocator);
  rclc_executor_add_subscription(&executor, &sub, &msg, &messageCb, ON_NEW_DATA);
}

void loop()
{
  lwheel.data = rmcs.Position_Feedback(slave_id3);  // the function reads the encoder value from the motor with slave id 4
  rwheel.data = -rmcs.Position_Feedback(slave_id1);  // the function reads the encoder value from the motor with slave id 1

  rcl_publish(&left_ticks_pub, &lwheel, NULL);   // publish left enc values
  rcl_publish(&right_ticks_pub, &rwheel, NULL);  // publish right enc values

  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));

  // If no new command has been received, stop the motors
  if (!new_command_received)
  {
    rmcs.Brake_Motor(slave_id1, 0);
    rmcs.Brake_Motor(slave_id2, 0);
    rmcs.Brake_Motor(slave_id3, 0);
    rmcs.Brake_Motor(slave_id4, 0);  // Stop the motors
    rmcs.Brake_Motor(slave_id1, 1);
    rmcs.Brake_Motor(slave_id2, 1);
    rmcs.Brake_Motor(slave_id3, 1);
    rmcs.Brake_Motor(slave_id4, 1);
  }
  else
  {
    new_command_received = false;  // Reset the flag after executing the command
  }
}