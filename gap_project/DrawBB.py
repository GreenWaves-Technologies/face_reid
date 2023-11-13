### Copyright (C) 2020 GreenWaves Technologies
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

#im = np.array(Image.open('input_rgb.ppm'), dtype=np.uint8)
im = np.array(Image.open('vga_samples/test_image_raw_bayer480x480_2.pgm'), dtype=np.uint8)
#im = np.array(Image.open('francesco_cropped_r256.ppm'), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

rect = patches.Rectangle((172.906494,295.751953),105.585938,105.585938,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
kp = patches.Circle((201.313477,318.724365),radius=1,color='green')
ax.add_patch(kp)
kp = patches.Circle((243.413086,325.239258),radius=1,color='green')
ax.add_patch(kp)
kp = patches.Circle((214.116211,343.183594),radius=1,color='green')
ax.add_patch(kp)
kp = patches.Circle((211.889648,368.408203),radius=1,color='green')
ax.add_patch(kp)
kp = patches.Circle((180.527344,332.153320),radius=1,color='green')
ax.add_patch(kp)
kp = patches.Circle((274.628906,345.703125),radius=1,color='green')
ax.add_patch(kp)


plt.savefig("OUTPUT.png")
################################

#plt.show()