# IGCN : Image-to-graph convolutional network 
IGCN is a learning framework for 2D/3D deformable model registration and alignment, and shape reconstruction from a single-viewpoint projection image. The generative network learns translation from the input projection image to a displacement map, and the GCN learns mesh deformation based on the sampled per-vertex feature and connectivity.


# Examples
https://user-images.githubusercontent.com/93433071/139565499-9b918d6a-a378-45d8-8419-3b7affbd4e22.mp4

- Left (input): digitally reconstructed radiograph images generated from 4D-CT data (10-frame sequential volumes)  
- Center (output): registered mesh of abdominal organs
- Right (error): target (magenta) and predicted (cyan) mesh  

# Prerequisites
- Python 3.9
- NVIDIA CUDA 11.2.0 and cuDNN 8.1.1
- TFLearn with Tensorflow backend

# Reference
If you use this code for your research, please cite

- IGCN (The latest version):  
M. Nakao, M. Nakamura, T. Matsuda, IGCN: Image-to-graph Convolutional Network for 2D/3D Deformable Registration, arXiv preprint, 2021.

- IGCN Warp (MICCAI version):   
M. Nakao, M. Nakamura, T. Matsuda, Image-to-Graph Convolutional Network for Deformable Shape Reconstruction from a Single Projection Image, International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), pp. 259-268, 2021.

