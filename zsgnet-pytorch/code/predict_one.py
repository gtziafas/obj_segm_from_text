from mdl import get_default_net
from evaluator import get_default_eval
import torch
import spacy
import numpy as np
import cv2
from yacs.config import CfgNode as CN
import json
from typing import Dict
import time
from pprint import pprint

nlp = spacy.load('en_core_web_md')
device='cpu'
if torch.cuda.is_available():
  device='cuda'

class PredictForROS():
	def __init__(self,cfg_path,model_path):
		cfg = CN(json.load(open(cfg_path)))
		self.cfg=cfg
		if type(cfg['ratios']) != list:
			self.ratios = eval(cfg['ratios'], {})
		else:
			self.ratios = cfg['ratios']
		if type(cfg['scales']) != list:
			self.scales = cfg['scale_factor'] * np.array(eval(cfg['scales'], {}))
		else:
			self.scales = cfg['scale_factor'] * np.array(cfg['scales'])

		self.cfg.device = device
		self.eval_fn = get_default_eval(self.ratios, self.scales, self.cfg)
		self.resized_img_size = self.cfg.resize_img
		self.phrase_len = 50
		self.zsg_net = get_default_net(num_anchors=9, cfg=cfg)

		self.device = torch.device(self.cfg.device)
		self.zsg_net.to(self.device)

		ck = torch.load(model_path)
		self.zsg_net.load_state_dict(ck['model_state_dict'], strict=False)
		self.zsg_net.eval()


	def create_query_embedding(self,q_chosen):
		q_chosen = q_chosen.strip()
		qtmp = nlp(str(q_chosen))
		if len(qtmp) == 0:
			# logger.error('Empty string provided')
			raise NotImplementedError
		qlens = len(qtmp)

		q_chosen = q_chosen + ' PD' * (self.phrase_len - qlens)
		q_chosen_emb = nlp(q_chosen)
		if not len(q_chosen_emb) == self.phrase_len:
			q_chosen_emb = q_chosen_emb[:self.phrase_len]

		q_chosen_emb_tensor = torch.from_numpy(np.array([q.vector for q in q_chosen_emb]))

		return q_chosen_emb_tensor, torch.tensor([qlens])

	def predict(self,img,query):
		#img = pil2tensor(img, np.float_).float().div_(255)
		#img = np.frombuffer(img.data,dtype=np.uint8).reshape(img.height,img.width,-1)
		h, w, c = img.shape
		img = cv2.resize(img, (self.resized_img_size[0],self.resized_img_size[1]))
		inp={}
		inp['img'] = torch.from_numpy(img).float().reshape(1,c,self.resized_img_size[0],self.resized_img_size[1]).to(self.device)
		inp['qvec'], inp['qlens'] = self.create_query_embedding(query)
		inp['qvec'] = inp['qvec'].float().reshape(1, inp['qvec'].shape[0], inp['qvec'].shape[1]).to(self.device)
		inp['qlens'] = inp['qlens'].to(self.device)
		inp['img_size'] = torch.tensor([h,w]).to(self.device)

		inp['annot'] = torch.rand(1,4).float().to(self.device) + 1

		output = self.zsg_net(inp)

		output = self.eval_fn.get_best_box(output,inp['img_size'])
		#output = self.eval_fn(output, inp) 

		return output
