<p align="center">
    <picture>
    <img alt="gaustudio" src="" width="50%">
    </picture>
</p>


<p align="center"><b>
gaustudio is a unified framework for 3D Gaussian Splatting and recent 3DGS-based work. The framework is launched as an opensource project by students in [GAP lab](https://gaplab.cuhk.edu.cn/) at [FNii](https://fnii.cuhk.edu.cn/) and [CUHK(SZ)](https://www.cuhk.edu.cn/en). 
</b></p>

## Installation
Before installing the software, please note that the following steps have been tested on Ubuntu 20.04. If you encounter any issues during the installation on Windows, we are open to addressing and resolving such issues.

### Prerequisites
* NVIDIA graphics card with at least 6GB VRAM
* CUDA installed
* Python >= 3.8

### Optional Step: Create a Conda Environment
It is recommended to create a conda environment before proceeding with the installation. You can create a conda environment using the following commands:
```sh
# Create a new conda environment
conda create -n gaustudio python=3.8
# Activate the conda environment
conda activate gaustudio
```

### Step 1: Install PyTorch
You will need to install PyTorch. The software has been tested with torch1.12.1+cu113 and torch2.0.1+cu118, but other versions should also work fine. You can install PyTorch using conda as follows:
```
# Example command to install PyTorch version 1.12.1+cu113
conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch

# Example command to install PyTorch version 2.0.1+cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install Dependencies
Install the necessary dependencies by running the following command:
```sh
pip install -r requirements.txt
```

### Optional Step: Install PyTorch3D
If you require mesh rendering and further mesh refinement, you can install PyTorch3D follow the [link](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md):
