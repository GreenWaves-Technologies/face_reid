import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from blazeface import BlazeFace
import math
import onnxruntime as ort
from scipy.spatial import distance
from numpy.linalg import norm
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


NORMALIZE_HISTOGRAM=False

def crop_and_resize(img, w, h):
	im_h, im_w, channels = img.shape
	res_aspect_ratio = w/h
	input_aspect_ratio = im_w/im_h

	if input_aspect_ratio > res_aspect_ratio:
		im_w_r = int(input_aspect_ratio*h)
		im_h_r = h
		img = cv2.resize(img, (im_w_r , im_h_r))
		x1 = int((im_w_r - w)/2)
		x2 = x1 + w
		img = img[:, x1:x2, :]
	if input_aspect_ratio < res_aspect_ratio:
		im_w_r = w
		im_h_r = int(w/input_aspect_ratio)
		img = cv2.resize(img, (im_w_r , im_h_r))
		y1 = int((im_h_r - h)/2)
		y2 = y1 + h
		img = img[y1:y2, :, :]
	if input_aspect_ratio == res_aspect_ratio:
		img = cv2.resize(img, (w, h))

	return img


def plot_detections(img, detections, with_keypoints=True):
	fig, ax = plt.subplots(1, figsize=(10, 10))
	ax.grid(False)
	ax.imshow(img)
	
	if isinstance(detections, torch.Tensor):
		detections = detections.cpu().numpy()

	if detections.ndim == 1:
		detections = np.expand_dims(detections, axis=0)

	print("Found %d faces" % detections.shape[0])
		
	for i in range(detections.shape[0]):
		ymin = detections[i, 0] * img.shape[0]
		xmin = detections[i, 1] * img.shape[1]
		ymax = detections[i, 2] * img.shape[0]
		xmax = detections[i, 3] * img.shape[1]

		rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
								 linewidth=1, edgecolor="r", facecolor="none", 
								 alpha=detections[i, 16])
		ax.add_patch(rect)

		if with_keypoints:
			for k in range(6):
				kp_x = detections[i, 4 + k*2    ] * img.shape[1]
				kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
				circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
										edgecolor="lightskyblue", facecolor="none", 
										alpha=detections[i, 16])
				ax.add_patch(circle)
		
	plt.show()

def get_coordinates(img,detections):
	if isinstance(detections, torch.Tensor):
		detections = detections.cpu().numpy()

	if detections.ndim == 1:
		detections = np.expand_dims(detections, axis=0)

	ymin = detections[0, 0] * img.shape[0]
	xmin = detections[0, 1] * img.shape[1]
	ymax = detections[0, 2] * img.shape[0]
	xmax = detections[0, 3] * img.shape[1]

	for k in range(2):
		kp_x = detections[0, 4 + k*2    ] * img.shape[1]
		kp_y = detections[0, 4 + k*2 + 1] * img.shape[0]
		#print(kp_x,kp_y)

	return xmin, ymin, xmax, ymax


def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result

def euclidean_distance(a, b):
	x1 = a[0]; y1 = a[1]
	x2 = b[0]; y2 = b[1]
	return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def get_face(img,detections,move):
	if isinstance(detections, torch.Tensor):
		detections = detections.cpu().numpy()

	if detections.ndim == 1:
		detections = np.expand_dims(detections, axis=0)

	ymin = detections[0, 0] * img.shape[0] + move 
	xmin = detections[0, 1] * img.shape[1] + move 
	ymax = detections[0, 2] * img.shape[0] + move 
	xmax = detections[0, 3] * img.shape[1] + move 
	

	## first is right eye (left for person who watch)
	## second is left eye (right for person who watch)

	# eyes = []
	# for k in range(2):
	# 	kp_x = detections[0, 4 + k*2    ] * img.shape[1]
	# 	kp_y = detections[0, 4 + k*2 + 1] * img.shape[0]
	# 	eyes.append([kp_x,kp_y])
	# 	#print(kp_x,kp_y)
	# #x_distance = np.abs(eyes[1][0]-eyes[0][0])
	# #y_distance = np.abs(eyes[1][1]-eyes[0][1])
	
	# left_eye_center = (eyes[0][0],eyes[0][1])
	# right_eye_center = (eyes[1][0],eyes[1][1])
	# if eyes[0][1] < eyes[1][1]:
	# 	point_3rd = (eyes[1][0], eyes[0][1])
	# 	direction = 1 #rotate same direction to clock
	# else:
	# 	point_3rd = (eyes[0][0], eyes[1][1])
	# 	direction = -1 #rotate inverse direction of clock

	# a = euclidean_distance(left_eye_center, point_3rd)
	# b = euclidean_distance(right_eye_center, left_eye_center)
	# c = euclidean_distance(right_eye_center, point_3rd)

	# cos_a = (b*b + c*c - a*a)/(2*b*c)
	# #print("cos(a) = ", cos_a)
 
	# angle = np.arccos(cos_a)
	# #print("angle: ", angle," in radian")
 
	# angle = (angle * 180) / math.pi
	
	# if direction == -1:
	# 	angle = - angle
	
	# #print("angle: ", angle," in degree")

	#distance = np.sqrt(np.power(x_distance,2)+np.power(y_distance,2)) 
	#print(x_distance,y_distance,distance)
	face_img = img[int(ymin):int(ymax),int(xmin):int(xmax)]

	# face_img = rotate_image(face_img,angle)

	# face_img = face_img[30:-30,30:-30]
	
	return face_img

def cos_sim(a,b):
	return 100*round(1 - (np.dot(a, b)/(norm(a)*norm(b))),4)


front_net = BlazeFace()
front_net.load_weights("blazeface.pth")
front_net.load_anchors("anchors.npy")
back_net = BlazeFace(back_model=True).to(gpu)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")

# Optionally change the thresholds:
front_net.min_score_thresh = 0.75
front_net.min_suppression_threshold = 0.3

#load shufflenet
model_path = "face_id.onnx"
session = ort.InferenceSession(model_path)



folder = "test_img_hd"
image_files = ["francesco_1.png","francesco_2.png","manuele_1.png"  ,"manuele_2.png"  ]

#folder = "test_img_vga"
#image_files = ["francesco_1.ppm","francesco_2.ppm"]

face_id = []

for input_file in image_files:
	print (input_file)
	single_face_id = []
	for move in [0]:
		input_img = cv2.imread(folder+"/"+input_file)
		input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
		img_res = crop_and_resize(input_img,128,128)
		img_crop = crop_and_resize(input_img,min(input_img.shape[0],input_img.shape[1]),min(input_img.shape[0],input_img.shape[1]))
		#img_crop = crop_and_resize(input_img,112,112)
		front_detections = front_net.predict_on_image(img_res)
		front_detections.shape
		#get_coordinates(img_crop,front_detections)
		face = get_face(img_crop,front_detections,move)
		face = crop_and_resize(face,112,112)


		if NORMALIZE_HISTOGRAM==True:
			# apply histogram equalization 
			ycrcb_img = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
			# equalize the histogram of the Y channel
			ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
			# convert back to RGB color-space from YCrCb
			face = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)


		# ycrcb_img = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
		# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		# ycrcb_img[:,:,0] = clahe.apply(ycrcb_img[:,:,0])
		# face = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)

		cv2.imwrite("cropped_faces/"+input_file+"_face_crop.png",face)
		#Loading 
		img_crop_to_faceid=(face.astype(np.float32))/(256.0)
		img_crop_to_faceid = img_crop_to_faceid.transpose(2, 0, 1)
		face_id_input=img_crop_to_faceid.reshape(1,3,112,112)
		# Run inference
		outputs = session.run(None, {"input": face_id_input})
		#print(outputs)
		single_face_id.append(outputs)
	face_id.append(np.mean(single_face_id, axis=0))
	


#print(repr(face_id[0]))
#print(repr(face_id[2]))

francesco = np.mean(face_id[0:1], axis=0)
manuele = np.mean(face_id[2:3], axis=0)

#np.save("face_ids/francesco",francesco)
#np.save("face_ids/manuele",manuele)

print("Distance francesco_1 vs francesco_2")
print(cos_sim(np.array(face_id[0]).reshape(128),np.array(face_id[1]).reshape(128)))
print("Distance manuele_1 vs manuele_2")
print(cos_sim(np.array(face_id[2]).reshape(128),np.array(face_id[3]).reshape(128)))

print("Distance francesco_1 vs manuele_1")
print(cos_sim(np.array(face_id[0]).reshape(128),np.array(face_id[2]).reshape(128)))
print("Distance francesco_1 vs manuele_2")
print(cos_sim(np.array(face_id[0]).reshape(128),np.array(face_id[3]).reshape(128)))

print("Distance francesco_2 vs manuele_1")
print(cos_sim(np.array(face_id[1]).reshape(128),np.array(face_id[2]).reshape(128)))

print("Distance francesco_2 vs manuele_2")
print(cos_sim(np.array(face_id[1]).reshape(128),np.array(face_id[3]).reshape(128)))

cv2.imwrite("output_face.png",face)

#plot_detections(img_crop, front_detections)