import os
BATCH_SIZE = 256
SAVE_FREQ = 10
TEST_FREQ = 5
TOTAL_EPOCH = 70

RESUME = ''
SAVE_DIR = './model'
MODEL_PRE = 'CASIA_ShuffleFaceNet_'


CASIA_DATA_DIR = (os.path.dirname(os.path.abspath(__file__)))+'/DATASETS/CASIA-WebFace_cropped/' #'/'CASIA-WebFace
LFW_DATA_DIR   = (os.path.dirname(os.path.abspath(__file__)))+'/DATASETS/lfw_funneled/'

GPU = 0