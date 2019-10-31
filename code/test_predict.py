from predict_one import PredictForROS

if __name__ == "__main__":
	conf_path='./configs/cfg.json'
	#conf['resize_img'] = [300,300]
	model_path ="../flickr30k_zsg_1.pth"
	p =PredictForROS(conf_path,model_path)