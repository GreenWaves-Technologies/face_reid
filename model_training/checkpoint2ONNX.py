#Convert Checkpoint to ONNX

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from core import model

#All these network were trained with depthwise convs
#ckpt = torch.load("model_trained/CASIA_ShuffleFaceNet_20230627_132100/070.ckpt")
#ckpt = torch.load("model_trained/CASIA_ShuffleFaceNet_20230720_152502/050.ckpt")
#ckpt = torch.load("model_trained/CASIA_ShuffleFaceNet_20230720_181909/070.ckpt")
#ckpt = torch.load("model_trained/CASIA_ShuffleFaceNet_20230721_144456/150.ckpt")
#dataset with images cropped 112x112 with face detector
#this is best
#ckpt = torch.load("model_trained/CASIA_ShuffleFaceNet_20230724_114153/070.ckpt")


#Shufflenet 0.5x
#Got back to normal Conv2d and PReLU
#ckpt = torch.load("model_trained/CASIA_ShuffleFaceNet_20230724_161807/050.ckpt")


#Shufflenet 0.5x
#Activation Relu6, normalization -128/128
#ckpt = torch.load("model_trained/CASIA_ShuffleFaceNet_20230725_083654/060.ckpt")
#Not good accuracy


#Shufflenet 0.5x
#Activation Relu6, normalization -127.5/128
#ckpt = torch.load("model_trained/CASIA_ShuffleFaceNet_20230725_114612/080.ckpt")

#################################################
#All previous were not using correct network architecture
#################################################

#Shufflenet 0.5x
#Activation Relu6, normalization -127.5/128
#ckpt = torch.load("model_trained/CASIA_ShuffleFaceNet_20230726_093332/060.ckpt")


#Shufflenet 0.5x
#Activation Relu6, normalization -128/128
#ckpt = torch.load("model_trained/CASIA_ShuffleFaceNet_20230726_135210/050.ckpt")

#Shufflenet 0.5x
#Activation Relu6, normalization /256
ckpt = torch.load("model/best/050.pt")

fake_input=torch.Tensor(1,3,112,112)


torch.onnx.export(ckpt,       # model being run
                  fake_input,               # model input (or a tuple for multiple inputs)
                  "shufflenet.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}},
                  verbose=False)



"""
## This is to load a ckpt which needs the model implementation to be exactly the same
net = model.ShuffleFaceNet()

net.load_state_dict(ckpt['net_state_dict'])

fake_input=torch.Tensor(1,3,112,112)

torch.onnx.export(net,       # model being run
                  fake_input,               # model input (or a tuple for multiple inputs)
                  "shufflenet.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}},
                    verbose=False)
"""