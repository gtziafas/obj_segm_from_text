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

def image_buffer():
  pub = rospy.Publisher('/image_buffer_from_path/RGB', Image, queue_size=10)
  rospy.init_node('image_buffer_from_path', anonymous=True)
  rate = rospy.Rate(30)
  cv_image = cv2.imread('src/object_segmentation_from_text/data/test.jpg', cv2.IMREAD_COLOR)
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
    image_buffer()
  except rospy.ROSInterruptException:
    pass

if __name__ == '__main__':
    main(sys.argv)