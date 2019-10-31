#!/usr/bin/env python3
from __future__ import print_function

import roslib
roslib.load_manifest('object_segmentation_from_text')
import rospy

from object_segmentation_from_text.srv import DoSomething

def handle_do_something(req):
	print('Doing something')
	return DoSomething(req.a + req.b)

def do_something_server():
	rospy.init_node('model_inference_server')
	s = rospy.Service('model_inference', DoSomething, handle_do_something)
	print('Ready for model inference')
	rospy.spin()

if __name__ == 'main':
	do_something_server()