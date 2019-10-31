from mdl import get_default_net
import torch
import spacy
import numpy as np
import cv2
from yacs.config import CfgNode as CN
import json

nlp = spacy.load('en_core_web_md')


class PredictForROS():
	def __init__(self,cfg_path,model_path):
		cfg = CN(json.load(open(cfg_path)))
		self.cfg=cfg
		self.cfg.device = 'cpu'
		self.img_size = self.cfg.resize_img
		self.phrase_len = 50
		self.zsg_net = get_default_net(num_anchors=9, cfg=cfg)
		self.device = torch.device(cfg.device)
		self.zsg_net.to(self.device)

		ck = torch.load(model_path)
		self.zsg_net.load_from_state_dict(ck['model_state_dict'])
		self.zsg_net.eval()


	def create_query_embedding(self,q_chosen):
		q_chosen = q_chosen.strip()
		qtmp = nlp(str(q_chosen))
		if len(qtmp) == 0:
			# logger.error('Empty string provided')
			raise NotImplementedError
		qlen = len(qtmp)

		q_chosen = q_chosen + ' PD' * (self.phrase_len - qlen)
		q_chosen_emb = nlp(q_chosen)
		if not len(q_chosen_emb) == self.phrase_len:
			q_chosen_emb = q_chosen_emb[:self.phrase_len]

		q_chosen_emb_tensor = torch.from_numpy(np.array([q.vector for q in q_chosen_emb]))

		return q_chosen_emb_tensor, torch.tensor(qlen)

	def predict(self,img,query):
		#img = pil2tensor(img, np.float_).float().div_(255)
		img = np.frombuffer(img.data,dtype=np.uint8).reshape(img.height,img.width,-1)
		img = cv2.resize(img, (self.img_size[0],self.img_size[1]))
		inp={}
		inp['img'] = torch.from_numpy(img).float().reshape(-1,self.img_size[0],self.img_size[1]).to(self.device)
		inp['qvec'],inp['qlen'] = self.create_query_embedding(query)

		output = self.zsg_net(inp)
		print(output)
