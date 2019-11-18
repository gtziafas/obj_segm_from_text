#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('object_segmentation_from_text')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def enchance_img(img):
  # apply addaptive histogram equalization in YUV space to enhance contrast
  clahe = cv2.createCLAHE(clipLimit = 2., tileGridSize=(8,8))
  yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
  y, u, v = cv2.split(yuv_image)
  y_equ = clahe.apply(y)
  yuv_image = cv2.merge((y_equ, u, v))
  
  return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

def image_buffer(path):
  pub = rospy.Publisher('/image_buffer_from_path/RGB', Image, queue_size=10)
  rospy.init_node('image_buffer_from_path', anonymous=True)
  rate = rospy.Rate(3) #3 Hz
  cv_image = cv2.imread(path, cv2.IMREAD_COLOR)
  #cv_image = enchance_img(cv_image)
  while not rospy.is_shutdown():
    log_str = 'buffering image from path %s' % rospy.get_time()
    rospy.loginfo(log_str)
    try:
      pub.publish(CvBridge().cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)
    rate.sleep()

def main(args):
  try:
    image_buffer(args[1])
  except rospy.ROSInterruptException:
    pass

if __name__ == '__main__':
    main(sys.argv)
