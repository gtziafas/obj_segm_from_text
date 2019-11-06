from predict_one import PredictForROS
import cv2

if __name__ == "__main__":
	conf_path='./configs/cfg.json'
	#conf['resize_img'] = [300,300]
	model_path ="./tmp/models/flickr30k_zsg_2.pth"
	query = 'a bridge'
	img = cv2.imread('./data/flickr30k/flickr30k_images/1160441615.jpg', cv2.IMREAD_COLOR)
	p =PredictForROS(conf_path,model_path)
	box = p.predict(img, query)['pred_boxes'].squeeze()
	img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,0xFF), thickness=2)
	cv2.imwrite('test__.jpg', img)


