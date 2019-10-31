#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('object_segmentation_from_text')
import sys
import rospy
from std_msgs.msg import String

def caption_buffer():
  pub = rospy.Publisher('/caption_buffer_from_console/caption', String, queue_size=10)
  rospy.init_node('caption_buffer_from_console', anonymous=True)
  rate = rospy.Rate(60)
  handle = raw_input('waiting for caption!: ')
  while not rospy.is_shutdown():
    log_str = 'buffering caption from console %s' % rospy.get_time()
    rospy.loginfo(log_str)
    try:
      pub.publish(handle)
    except CvBridgeError as e:
      print(e)
    rate.sleep()

def main(args):
  try:
    caption_buffer()
  except rospy.ROSInterruptException:
    pass

if __name__ == '__main__':
    main(sys.argv)