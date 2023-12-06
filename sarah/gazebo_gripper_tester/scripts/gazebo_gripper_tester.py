#!/usr/bin/env python

import sys
import rospy
from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose
import tf

def raise_arms(coef):
  try:
    set_force=rospy.ServiceProxy("/gazebo/apply_joint_effort", ApplyJointEffort)
    
    time=rospy.Time()
    duration=rospy.Duration.from_sec(-1)
    
    # Raise all robots arms
    res=set_force("simple_manipulator1::arm_gripper_joint", coef * 200.0, time, duration)
    res=set_force("simple_manipulator2::arm_gripper_joint", coef * 200.0, time, duration)
    res=set_force("simple_manipulator3::arm_gripper_joint", coef * 200.0, time, duration)
  
  except rospy.ServiceException as e:
    print ("Service call failed %s"%e)



def open_grippers(coef):
  try:
    set_force=rospy.ServiceProxy("/gazebo/apply_joint_effort", ApplyJointEffort)
    
    
    time=rospy.Time()
    duration=rospy.Duration.from_sec(-1)
    
    
    # Open all grippers
    # Manipulator1
    res=set_force("simple_manipulator1::simple_gripper1::palm_left_finger", coef * 10.0, time, duration)
    res=set_force("simple_manipulator1::simple_gripper1::palm_right_finger", coef * -10.0, time, duration)
    res=set_force("simple_manipulator1::simple_gripper1::left_finger_tip_joint", coef * 5.0, time, duration)
    res=set_force("simple_manipulator1::simple_gripper1::right_finger_tip_joint", coef * -5.0, time, duration)
    
    # Manipulator2
    res=set_force("simple_manipulator2::simple_gripper2::palm_left_finger", coef * 10.0, time, duration)
    res=set_force("simple_manipulator2::simple_gripper2::palm_right_finger", coef * -10.0, time, duration)
    res=set_force("simple_manipulator2::simple_gripper2::left_finger_tip_joint", coef * 5.0, time, duration)
    res=set_force("simple_manipulator2::simple_gripper2::right_finger_tip_joint", coef * -5.0, time, duration)
    
    # Manipulator3
    res=set_force("simple_manipulator3::simple_gripper3::palm_left_finger", coef * 10.0, time, duration)
    res=set_force("simple_manipulator3::simple_gripper3::palm_right_finger", coef * -10.0, time, duration)
    res=set_force("simple_manipulator3::simple_gripper3::left_finger_tip_joint", coef * 5.0, time, duration)
    res=set_force("simple_manipulator3::simple_gripper3::right_finger_tip_joint", coef * -5.0, time, duration)
    res=set_force("simple_manipulator3::simple_gripper3::palm_front_finger", coef * -20.0, time, duration)
    res=set_force("simple_manipulator3::simple_gripper3::palm_rear_finger", coef * 20.0, time, duration)
    res=set_force("simple_manipulator3::simple_gripper3::front_finger_tip_joint", coef * -10.0, time, duration)
    res=set_force("simple_manipulator3::simple_gripper3::rear_finger_tip_joint", coef * 10.0, time, duration)
    
  
  except rospy.ServiceException as e:
    print ("Service call failed %s"%e)
    
    
def spawn_object(object_name, x, y, z, roll, pitch, yaw):
  try:
    spawn_object=rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    
    # get model
    f = open('/home/user_adm/.gazebo/models/' + object_name + '/model.sdf','r')
    sdff = f.read()
    
    # define pose
    pose=Pose()
    pose.position.x=x
    pose.position.y=y
    pose.position.z=z

    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]
    
    
    # Spawn object
    res=spawn_object(object_name + "2", sdff, "", pose, "")
    
    pose.position.y = 2
    
    res=spawn_object(object_name + "1", sdff, "", pose, "")
    
    pose.position.y = -2
    
    res=spawn_object(object_name + "3", sdff, "", pose, "")
    
  except rospy.ServiceException as e:
    print ("Service call failed %s"%e)


def delete_object(object_name):
  try:
    delete_object=rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    
   
    # Delete object
    res=delete_object(object_name + "1")
    res=delete_object(object_name + "2")
    res=delete_object(object_name + "3")
    
  except rospy.ServiceException as e:
    print ("Service call failed %s"%e)
    
def collect_result(object_name):

  try:
    get_model=rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    
    res = get_model(object_name + "1", "")
    
    success = False
    if res.pose.position.z > 0.1 :
      success = True
      
    print ("object " + object_name + " grabb with gripper 1 ", success)


    res = get_model(object_name + "2", "")
    
    success = False
    if res.pose.position.z > 0.1 :
      success = True
      
    print ("object " + object_name + " grabb with gripper 2 ", success)
    
    res = get_model(object_name + "3", "")
    
    success = False
    if res.pose.position.z > 0.1 :
      success = True
    
    
    print ("object " + object_name + " grabb with gripper 3 ", success)
   
  except rospy.ServiceException as e:
    print ("Service call failed %s"%e)
    
def test_object(object_name, x, y, z, roll, pitch, yaw):

  raise_arms(1.0)
  open_grippers(1.0)
  
  rospy.sleep(2.0)
  
  # spawn object
  spawn_object(object_name, x, y, z, roll, pitch, yaw)
  
  # Go down to gripp
  raise_arms(-1.0)
  
  rospy.sleep(2.0)
  
  # grip object
  open_grippers(-3.0)
  
  rospy.sleep(2.0)
  
  # raise arm
  raise_arms(1.0)
  
  rospy.sleep(3.0)
  
  # collect object position
  collect_result(object_name)
  
  rospy.sleep(2.0)
  
  # Remove object
  delete_object(object_name)
  
  # Release forces
  open_grippers(2.0)
  
  raise_arms(-1.0)
  
  
    
if __name__=="__main__":
  rospy.wait_for_service("/gazebo/apply_joint_effort")
  rospy.wait_for_service("/gazebo/spawn_sdf_model")
  rospy.wait_for_service("/gazebo/delete_model")
  rospy.wait_for_service("/gazebo/get_model_state")
  
  test_object("beer", 0.72, 0, 0, 0, 0, 0)
  test_object("wood_cube_5cm", 0.72, 0, 0, 0, 0, 0)
  test_object("irobot_hand", 0.72, 0, 0, 0, 0, 0)
  
  
  
  
