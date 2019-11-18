#!/usr/bin/env python3
"""
ROS Node for masking RGB-D data and wrapping camera info data for PointCloud2 extraction
Author: Giorgos Tziafas
"""
from __future__ import print_function

# import ROS utils
import roslib
roslib.load_manifest('object_segmentation_from_text')
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import sys

class _3DImageBuffer():
	def __init__(self):
		rospy.init_node('masked_buffer_for_3d', anonymous=True)

		# init subs/pubs and callback functions
		self.rgb_pub = rospy.Publisher('/object_segmentation_from_text/RGB_masked', Image, queue_size=10)
		self.depth_pub = rospy.Publisher('/object_segmentation_from_text/depth_masked', Image, queue_size=10)
		self.camera_pub = rospy.Publisher('/object_segmentation_from_text/camera_info', CameraInfo, queue_size=10)
		self.rate = rospy.Rate(3) # @3 Hz

		self.depth_sub = rospy.Subscriber("/camera/depth_registered/hw_registered/image_rect", Image, self.got_depth)
		self.rgb_sub = rospy.Subscriber("/object_segmentation_from_text/RGB_msked", Image, self.got_rgb)
		self.camera_sub = rospy.Subscriber("/camera/rgb/camera_info", CameraInfo, self.got_info)

	def got_info(self, info):
		self.info = info 

	def got_depth(self, depth):
		self.depth = depth
		#print('depth', depth.header)

	def got_rgb(self, rgb):
		self.rgb = rgb
		#print('rgb', rgb.header)
		self.repub()

	def repub(self):
		self.depth.header = self.rgb.header
		self.info.header = self.rgb.header
	
		# re-publish data
		log_str = 'buffering masked images%s' % rospy.get_time()
		rospy.loginfo(log_str)
		self.rgb_pub.publish(self.rgb)
		self.depth_pub.publish(self.depth)
		self.camera_pub.publish(self.info)
		self.rate.sleep()

def main(args):
  buf = _3DImageBuffer()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
