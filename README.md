# Multiple Arducam Camera System on a Jetson Orin
Research @ LEMS, Brown University <br />

This repository maintains the code of the experiments when working with multiple Arducam cameras on a Jetson Orin. Currently, a [Arducam Quadrascopic camera bundle kit](https://www.arducam.com/arducam-1mp4-quadrascopic-camera-bundle-kit-for-raspberry-pi-nvidia-jetson-nano-xavier-nx-four-ov9281-global-shutter-monochrome-camera-modules-and-camarray-camera-hat.html) (Arducam Jetvariety OV9281 cameras) is adopted to stream 4 synchronized HD images at 45 FPS to a Jetson Orin Nano, which can be extended to two quadrascopic cameras to stream 8 (almost synchronized) HD images at 45 FPS. 

## Streaming 8 images
The python code `arducam_pair_displayer.py` demonstrates how eight images are read using two quadrascopic cameras connected to the two CSI-MIPI interfaces on the Jetson side. Pressing `g` with root previlege enables saving 8 images to a specified output directory.

## Tasks applied to the captured images
### Detect and match SIFT features
The code `arducam_cams_popsift_match.py` hosts the highest level interface for streaming 4 cameras from the Quadrascopic bundle kit. Basically, it accepts four images at the same time, stacks and passes them to the pybind function [`popsift_match.match_multiple_pairs_from_arrays`](https://github.com/C-H-Chien/multi_cam_jetson/blob/4eafd09b767966a75ea5a35a1e2e830135ac8aa9/arducam_cams_popsift_match.py#L122).
SIFT features of the four images captured at different times are compared to construct SIFT correspondences. 
The returned data of the pybind are the breakdown timings of each step required for a complete SIFT feature detection and matching, which involves initialization, SIFT keypoint extraction time, matching time, cleanup, _etc_. <br />

The pybind function `popsift_match.match_multiple_pairs_from_arrays` connects the [`match_multiple_pairs_from_arrays`](https://github.com/C-H-Chien/multi_cam_jetson/blob/4eafd09b767966a75ea5a35a1e2e830135ac8aa9/popsift/src/application/py_match.cpp#L718)
from [`py_match.cpp`](https://github.com/C-H-Chien/multi_cam_jetson/blob/main/popsift/src/application/py_match.cpp) which runs the SIFT keypoint extraction and matching on GPU.

## Acknowledgement
- Code for displaying images from Arducam cameras are based on [Arducam example code](https://github.com/ArduCAM/MIPI_Camera). <br />
- GPU SIFT keypoint detection and matching code are based on [PopSIFT](https://github.com/alicevision/popsift). See `README` under the `popsift` folder.
