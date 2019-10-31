#!/usr/bin/env python3
from __future__ import print_function

import roslib
roslib.load_manifest('object_segmentation_from_text')
import sys
import rospy
import numpy as np 
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image

import torch
import spacy 
nlp = spacy.load('en_core_web_md')

device='cpu'
if torch.cuda.is_available():
  device='cuda'
rospy.loginfo("Using device=%s\n" % device)

# ros param ?
img_size = 300

class zsg_network_data_preparation():

  def __init__(self):
    self.image_pub = rospy.Publisher("/processed/RGB",Image, queue_size=10)
    rospy.init_node('zsg_network_data_preparation', anonymous=True)
    self.rate = rospy.Rate(30)

    self.caption = ""
    self.caption_sub = rospy.Subscriber("/caption_buffer_from_console/caption",String, self.caption_callback)
    self.image_sub = rospy.Subscriber("/image_buffer_from_path/RGB",Image, self.image_callback)
    #self.publish_processed_image(self.image)
    
  def caption_callback(self, caption):
    self.caption = str(caption.data) 

    #convert to sequence of spacy tensor embeddings of 300-d 
    doc = spacy.tokens.doc.Doc(nlp.vocab, words=self.caption.split())
    doc_tensor = torch.tensor([word.vector for word in doc]).long().to(device)
    print('doc_tensor: {}'.format(doc_tensor.shape))

  def image_callback(self, image):
    self.image = image 

    # convert to num_channels X height X widht 3d image tensor
    image_np = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    image_np = cv2.resize(image_np, (img_size,img_size))
    image_np = cv2.rectangle(image_np, (50,50), (150,150), (0,0,255), 2)
    image_tensor = torch.from_numpy(image_np).float().reshape(3,img_size,img_size).to(device)
    print('image tensor: {}'.format(image_tensor.shape))
    #print('image numpy: {}'.format(image_np.shape))

  def process_image(self, image):
    # draw a random box 
    image = cv2.rect(image, (50,50), (150,150), color=(0,0,255), thickness=2)
    return image

  def publish_processed_image(self, image):
    while not rospy.is_shutdown():
      log_str = 'publishing processed frame %s' % rospy.get_time()
      rospy.loginfo(log_str)
      self.image_pub.publish(image)
      self.rate.sleep()

def main(args):
  pp = zsg_network_data_preparation()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)