# IGCN : Image-to-graph network for shape reconsruction from a single-viewpoint projection image
IGCN is designed as a generalized 2D/3D deformable mesh registration framework for shape reconstruction from a single-viewpoint projection image. 


# Examples
https://user-images.githubusercontent.com/93433071/139565499-9b918d6a-a378-45d8-8419-3b7affbd4e22.mp4

- Left (input): 10-frame sequential digitally reconstructed radiograph images from 4D-CT data 
- Center (output): registered meshes of abdominal organs (<span style="color: red;">liver</span>, blue: stomach, green: duodenum, yellow: kidney and gray: pancreatic cancer)  
- Right (error): target (magenta) and predicted (cyan) mesh  

# Prerequisites
- Python 3.9
- NVIDAI CUDA 11.2.0 and cuDNN 8.1.1
- TFLearn with Tensorflow backend

# Reference
If you use this code for your research, please cite:

- IGCN


- IGCN Warp (MICCAI version): 

M. Nakao, M. Nakamura, T. Matsuda, "Image-to-Graph Convolutional Network for Deformable Shape Reconstruction from a Single Projection Image", International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), pp. 259-268, 2021.

