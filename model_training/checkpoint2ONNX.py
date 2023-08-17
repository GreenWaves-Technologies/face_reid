#Convert Checkpoint to ONNX

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from core import model

#Shufflenet 0.5x
#Activation Relu6, normalization /256
ckpt = torch.load("model/best/050.pt")

fake_input=torch.Tensor(1,3,112,112)

torch.onnx.export(ckpt.cpu(),       # model being run
                  fake_input,               # model input (or a tuple for multiple inputs)
                  "../model/shufflenet.onnx",   # where to save the model (can be a file or file-like object)
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