import numpy as np
import imageio
import torch
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
sys.path.append("..")
# from retrieval.dataloaders.preprocessing import preprocess


def crop_center(img,cropx,cropy,offset_x,offset_y):
    z,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startx = startx + offset_x
    starty = starty + offset_y
    return img[:,starty:starty+cropy,startx:startx+cropx]


class LFW(object):
    def __init__(self, imgl, imgr):

        self.imgl_list = imgl
        self.imgr_list = imgr

    def __getitem__(self, index):
        imgl = imageio.imread(self.imgl_list[index])
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)
        imgr = imageio.imread(self.imgr_list[index])
        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)

        # imgl = imgl[:, :, ::-1]
        # imgr = imgr[:, :, ::-1]
        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            ycrcb_img = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2YCrCb)
            # equalize the histogram of the Y channel
            ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
            # convert back to RGB color-space from YCrCb
            imglist[i]  = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
 
            #imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = (imglist[i]) / 256.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
            decenter_x = np.random.choice(5)*2-1
            decenter_y = np.random.choice(5)*2-1
            imglist[i] = crop_center(imglist[i],112,112,decenter_x,decenter_y)
            
            # apply histogram equalization
            # imgplot = plt.imshow(imglist[i].transpose(1, 2, 0))
            # plt.show()
            #for debug purposes

        imgs = [torch.from_numpy(i).float() for i in imglist]
        return imgs

    def __len__(self):
        return len(self.imgl_list)


if __name__ == '__main__':
    data_dir = '/home/francesco/works/machine_learning/face_id/DATASETS/lfw_funneled/'
    from lfw_eval import parseList
    nl, nr, folds, flags = parseList(root=data_dir)
    dataset = LFW(nl, nr)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)
