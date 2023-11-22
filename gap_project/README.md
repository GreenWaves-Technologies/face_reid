# Face Re-Identification on Gap9

This project shows Face Detection + Face Identification on Gap9. The project is composed of two Neural Networks running on Gap9 and some ISP features. The following pictures resumes the full processing:

<p align="center">
  <img src="readme_images/Face_REID_Algo.png" alt="Face Reid Algo" />
</p>


## Gap Project Running Modes:

Initialize the build directory and open the cmake menuconfig:

```
# Init cmake build directory, named "build"
cmake -B build
# Configure the application, using build directory "build"
# --> here you can choose running modes and all SDK settigns
cmake --build build --target menuconfig
```

In just opened menuconfig you can follow the following path `FaceID Menu --> Application Mode` to choose between 4 different modes:

1. Only face id Inference Mode : launch the model on specific input images
2. Face Detection + Face Id Inference Mode: launch the 2 models on specific input images
3. Generate Signature of Face with Camera
4. Demo Mode: use the camera

### Mode 1 and 2

The first two modes execute the project with input images from PC. They can be run in GVSOC or board. They are also used to check non-regressions due to SDK and tools updates.

### Mode 3 - Generate Signature of Face with Camera


**Attention: This mode can only be run on target board!**
To select it you can go to menuconfig `GAP_SDK --> Platform` and select board.

### Mode 4 - Demo Mode: use the camera



**Attention: This mode can only be run on target board!**
To select it you can go to menuconfig `GAP_SDK --> Platform` and select board.
--------------

Finally to run you can execute this command:

```
# Run the target
cmake --build build --target run
```
