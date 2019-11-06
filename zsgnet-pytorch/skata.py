import cv2

img = cv2.imread('./data/flickr30k/flickr30k_images/1160441615.jpg', cv2.IMREAD_COLOR)
img = cv2.rectangle(img, (143,106), (356,315), 0xFF, thickness=2)
img = cv2.rectangle(img, (99,117), (385,377), (0,0,0xFF), thickness=2)
cv2.imshow('eleos', img)
while True:
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()


