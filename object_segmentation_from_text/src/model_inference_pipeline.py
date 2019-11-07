#!/usr/bin/env python3
"""
ROS Node for data preparation, model loading, inference and processed publishing
Author: Giorgos Tziafas
"""
from __future__ import print_function

# import ROS utils
import roslib
roslib.load_manifest('object_segmentation_from_text')
import rospy
import message_filters as mf 
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# import Python/PyTorch utils
import sys
import torch
import cv2
import time 
import json
import numpy as np
from pathlib import Path 
from typing import Dict 
from functools import partial 
from yacs.config import CfgNode as CN
from anchors import (create_anchors, reg_params_to_bbox, x1y1x2y2_to_y1x1y2x2)

# import spacy module for GloVe pre-trained word embeddings
import spacy 
nlp = spacy.load('en_core_web_md')

# load ZSG network module 
from ZSGNetwork import get_default_net

# use CUDA tensors if available
device='cpu'
if torch.cuda.is_available():
  device='cuda' 

# ros params??
resize_image = 300
phrase_len = 50
cfg_path = '/home/ggtz/vis_gr_ws/src/object_segmentation_from_text/configs/cfg.json'
model_path = '/home/ggtz/vis_gr_ws/src/object_segmentation_from_text/checkpoints/zsgnet_flickr30k_best.pth'

# collate function for pushing tensor objects into batches 
def collater(batch):
    qlens = torch.Tensor([i['qlens'] for i in batch])
    max_qlen = int(qlens.max().item())
    out_dict = {}
    for k in batch[0]:
        out_dict[k] = torch.stack([b[k] for b in batch]).float()
    out_dict['qvec'] = out_dict['qvec'][:, :max_qlen]

    return out_dict

# bounding box reshaping
def reshape(box, new_size):
  """
  box: (N, 4) in y1x1y2x2 format
  new_size: (N, 2) stack of (h, w)
  """
  box[:, :2] = new_size * box[:, :2]
  box[:, 2:] = new_size * box[:, 2:]
  return box

# covert cv2 style image to torch image tensor
def cv2_to_tensor(image, dtype):
  a = np.asarray(image)
  if a.ndim == 2: # make it work for grayscale also
      a = np.expand_dims(a, 2)
  a = np.transpose(a, (1, 0, 2))
  a = np.transpose(a, (2, 1, 0))

  return torch.from_numpy(a.astype(dtype, copy=False)).float().div_(0xFF).to(device)

# the model in inference mode with all ROS utils and data preparations from config file
class ModelInferencePipeline():
  def __init__(self, cfg_path=cfg_path, model_path=model_path):
    rospy.init_node('model_inference_pipeline', anonymous=True)
    # ros params ??
    self.resize_image = resize_image
    self.phrase_len = phrase_len
    self.device = device
    rospy.loginfo('Using torch.device={}.'.format(self.device))

    # load network
    self.cfg = CN(json.load(open(cfg_path)))
    self.mpath = model_path
    self.mfile = Path(model_path)
    self.load_network()
    
    # init subs/pubs and callback functions
    self.rgb1_pub = rospy.Publisher('/object_segmentation_from_text/RGB_cropped', Image, queue_size=10)  # RGB image publisher
    self.rgb2_pub = rospy.Publisher('/object_segmentation_from_text/RGB_with_box', Image, queue_size=10) # -""---""---""----
    self.depth_pub = rospy.Publisher('/object_segmentation_from_text/depth_cropped', Image, queue_size=10) # Depth image publisher
    self.rate = rospy.Rate(5) # @5 Hz

    self.caption = ' PD' # PAD value given temporarily
    self.cached = self.caption # to dump previous captions
    self.caption_sub = rospy.Subscriber("/caption_buffer_from_console/caption", String, self.got_caption)
    self.rgb_sub = rospy.Subscriber("/image_buffer_from_path/RGB", Image, self.got_img)
    #self.rgb_sub = mf.Subscriber("/camera/rgb/image_rect_color", Image)
    #self.depth_sub = mf.Subscriber("/camera/depth_registered/hw_registered/image_rect", Image)
    #self.ats = mf.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=5, slop=.1)
    #self.ats.registerCallback(self.got_imgs)

  def load_network(self):

    # anchors for bbox proposals
    if type(self.cfg['ratios']) != list:
      self.ratios = eval(self.cfg['ratios'], {})
    else:
      self.ratios = self.cfg['ratios']
    if type(self.cfg['scales']) != list:
      self.scales = self.cfg['scale_factor'] * np.array(eval(self.cfg['scales'], {}))
    else:
      self.scales = self.cfg['scale_factor'] * np.array(self.cfg['scales'])
    self.anchs = None
    self.get_anchors = partial(
            create_anchors, ratios=self.ratios,
            scales=self.scales, flatten=True)
    self.num_anchors = len(self.ratios) * len(self.scales)

    # init model
    self.model = get_default_net(num_anchors=self.num_anchors, cfg=self.cfg).to(self.device)

    # load weights from pretrained model path
    if not self.mfile.exists():
      rospy.loginfo('No existing model in {}'.format(self.mfile))
      return
    try:
      checkpoint = torch.load(open(self.mfile, 'rb'))
      rospy.loginfo('Loaded model from {} Correctly'.format(self.mfile))
    except OSError as e:
      rospy.loginfo('Some problem with model path: {}. Exception raised {}'.format(self.path, e))
      raise e

    if self.cfg['load_normally']:
      self.model.load_state_dict(checkpoint, strict=self.cfg['strict_load']) 
    self.model.eval()

  def got_caption(self, caption):
    # save buffed string 
    self.caption = str(caption.data)

    # create word embeddings for input caption sequence
    self.caption_embdds, self.caption_len = self.create_caption_embeddings(self.caption)

  def got_img(self, img):
    self.h, self.w, self.img_step = img.height, img.width, img.step
    self.rgb = np.frombuffer(img.data, dtype=np.uint8).reshape(self.h, self.w, -1)
    img = cv2.resize(self.rgb, (self.resize_image, self.resize_image))
    self.img = cv2_to_tensor(img, np.float_)

    # predict box and publish processed data 
    self.inference()
    self.cached = self.caption
    self.publish_processed_image()

  def got_imgs(self, rgb, depth):
    print('IN')
    # convert RGB sensor_msgs.msgs/Image byte array to numpy array 
    self.h, self.w = rgb.height, rgb.width
    self.rgb = np.frombuffer(rgb.data, dtype=np.uint8).reshape(self.h, self.w, -1)

    # resize to desired image size and convert to torch tensor
    self.img = cv2.resize(self.rgb, (self.resize_image, self.resize_image))
    self.img = cv2_to_tensor(self.img, np.float_)

    # save buffed depth image 
    self.depth = np.frombuffer(depth.data, dtype=np.uint8).reshape(self.h, self.w, -1)

  # convert given caption string of length qlen to torch tensor of shape [qlen X 300]
  def create_caption_embeddings(self, q):
    q = q.strip()
    qtmp = nlp(str(q))
    if not len(qtmp):
      raise NotImplementedError
    qlen = len(qtmp)

    q += ' PD' * (self.phrase_len - qlen)
    q_emb = nlp(q)
    if not len(q_emb) == self.phrase_len:
      q_emb = q_emb[:self.phrase_len]
    q_emb_tensor = torch.from_numpy(np.array([q.vector for q in q_emb]))

    return q_emb_tensor.float().to(self.device), torch.tensor(qlen).float().to(self.device)

  def inference(self):
    # prepare input dict for ZSG net and go to inference
    self.inp = {
        'img'       :  self.img,
        'qvec'      :  self.caption_embdds,
        'qlens'     :  self.caption_len,
        'idxs'      :  torch.tensor(0.).to(self.device),
        'img_size'  :  torch.tensor([self.h, self.w]).to(self.device)
    }
    self.inp = collater([self.inp])

    with torch.no_grad():
      # forward pass
      out = self.model(self.inp)

      # get bbox
      box_dict = self.get_bbox(out, self.inp['img_size'])

    self.bbox = box_dict['pred_box'].squeeze()
    #rospy.loginfo('Infered bounding box={}'.format(self.bbox))

  def get_bbox(self, out, img_size):
    att_box = out['att_out']
    reg_box = out['bbx_out']
    feat_sizes = out['feat_sizes']
    num_f_out = out['num_f_out']

    device = att_box.device

    if len(num_f_out) > 1:
        num_f_out = int(num_f_out[0].item())
    else:
        num_f_out = int(num_f_out.item())

    feat_sizes = feat_sizes[:num_f_out, :]

    if self.anchs is None:
      feat_sizes = feat_sizes[:num_f_out, :]
      anchs = self.get_anchors(feat_sizes)
      anchs = anchs.to(self.device)
      self.anchs = anchs
    else:
      anchs = self.anchs

    att_box_sigmoid = torch.sigmoid(att_box).squeeze(-1)
    att_box, ids_to_use = att_box_sigmoid.max(1)
    actual_bbox = reg_params_to_bbox(anchs, reg_box)

    best_box = torch.gather(actual_bbox, 1, ids_to_use.view(-1, 1, 1).expand(-1, 1, 4))
    best_box = best_box.view(best_box.size(0), -1)
    
    reshaped_box = x1y1x2y2_to_y1x1y2x2(reshape(
      (best_box + 1) / 2, img_size))

    box_dict = {
        'pred_box'    : reshaped_box,
        'pred_score'  : att_box
    }

    return box_dict

  def publish_processed_image(self):
    # draw box and caption string
    self.rgb = cv2.rectangle(self.rgb, (self.bbox[0], self.bbox[1]), (self.bbox[2],self.bbox[3]), (0,0,0xFF), 2)
    self.rgb = cv2.putText(self.rgb, self.caption, (self.bbox[0]+2, self.bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0xFF), thickness=2)

    #construct sensor_msgs/Image object
    img = Image()
    img.header.stamp = rospy.Time()
    img.header.frame_id = 'camera_rgb_optical_frame'
    img.height, img.width = self.h, self.w 
    img.encoding = 'bgr8'
    img.is_bigendian = 0
    img.step = self.img_step
    img.data = self.rgb.tobytes()

    # publish frame
    log_str = 'publishing processed frames %s' % rospy.get_time()
    rospy.loginfo(log_str)
    self.rgb2_pub.publish(img)   
    self.rate.sleep() 

  def publish_processed_images(self):
    # crop bbox data in both frames through masking
    msk = np.zeros_like(self.depth)
    msk = cv2.rectangle(msk, (self.bbox[0], self.bbox[1]), (self.bbox[2],self.bbox[3]), 0xFF, -1)
    rgb_proc = np.zeros_like(self.rgb)
    depth_proc = np.zeros_like(self.depth)
    rgb_proc[msk==0xFF, :] = self.rgb[msk==0xFF, :]
    depth_proc[msk==0xFF] = self.depth[msk==0xFF]

    # also publish the original frame with the box drawn
    rgb_with_box = cv2.rectangle(self.rgb, (self.bbox[0], self.bbox[1]), (self.bbox[2],self.bbox[3]), (0,0,0xFF), 2)

    # convert numpy arrays back to byte arrays for publishing
    rgb_proc = np.getbuffer(rgb_proc)
    depth_proc = np.getbuffer(depth_proc)

    # publish frames
    while not rospy.is_shutdown():
      log_str = 'publishing processed frames %s' % rospy.get_time()
      rospy.loginfo(log_str)
      self.rgb1_pub.publish(rgb_proc)
      self.rgb2_pub.publish(rgb_with_box)
      self.depth_pub.publish(depth_proc)
      self.rate.sleep()

def main(args):
  pp = ModelInferencePipeline()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)