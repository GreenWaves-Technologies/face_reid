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

im = np.array(Image.open('vga_samples/test_image_raw_bayer480x480.pgm'), dtype=np.uint8)
#im = np.array(Image.open('francesco_cropped_r256.ppm'), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

rect = patches.Rectangle((184.671021,80.141602),85.371094,85.371094,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
kp = patches.Circle((215.961914,109.584961),radius=1,color='green')
ax.add_patch(kp)
kp = patches.Circle((242.607422,107.285156),radius=1,color='green')
ax.add_patch(kp)
kp = patches.Circle((234.624023,124.453125),radius=1,color='green')
ax.add_patch(kp)
kp = patches.Circle((234.448242,139.013672),radius=1,color='green')
ax.add_patch(kp)
kp = patches.Circle((193.652344,119.904785),radius=1,color='green')
ax.add_patch(kp)
kp = patches.Circle((253.081055,113.415527),radius=1,color='green')
ax.add_patch(kp)




plt.savefig("OUTPUT.png")
################################

#plt.show()