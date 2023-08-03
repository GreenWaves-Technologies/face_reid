import numpy as np
import imageio
import os
import cv2
from sklearn import preprocessing
import torch
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("..")

def crop_center(img,cropx,cropy,offset_x,offset_y):
    z,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startx = startx + offset_x
    starty = starty + offset_y
    return img[:,starty:starty+cropy,startx:startx+cropx]

class CASIA_Face(object):
    def __init__(self, root):
        self.image_list = []
        self.label_list = []

        for r, _, files in os.walk(root):
            for f in files:
                self.image_list.append(os.path.join(r, f))
                self.label_list.append(os.path.basename(r))

        le = preprocessing.LabelEncoder()
        self.label_list = le.fit_transform(self.label_list)
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = imageio.imread(img_path)
        #img = np.resize(img, (112, 112)) #this is already done in NN forward
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        # convert back to RGB color-space from YCrCb
        img  = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)


        flip = np.random.choice(2)*2-1
        img = img[:, ::flip, :]
        

        #img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1) #Channel last to channel first

        #decenter_x = np.random.choice(4)*2-1
        #decenter_y = np.random.choice(4)*2-1
        # Commented out since images are already cropped with a face detector
        #img = crop_center(img,112,112,decenter_x,decenter_y)
        img = torch.from_numpy(img.copy())
        #img = img.add(-127.5).div(128.0).float()
        img = img.div(256).float()

        #img = torch.Tensor(3,250,250)

        return img, target

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    data_dir = '/home/francesco/works/machine_learning/face_id/DATASETS/CASIA-WebFace/'
    #data_dir = '/home/users/matheusb/recfaces/datasets/CASIA-WebFace/'
    dataset = CASIA_Face(root=data_dir)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)
