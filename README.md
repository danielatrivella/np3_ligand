# NP³ Ligand

---------------------------------------

This repository stores the NP³ projects for ligand interpretation in X-ray protein crystallography. It contains three modules:

- **np3_LigPCDS**: the repository of the LigPCDS project with the code to create the labeled dataset of 3D representations of ligand images in point cloud and the stratified training dataset.
- **np3_DL_segmentation**: the repository with the training pipeline for semantic segmentation tasks used to validate LigPCDS and obtain the DL models.
- **np3_blob_label**: repository of the NP³ Blob Label application &copy;

The NP³ is a project from the Drug Discovery Division ([DDD](https://lnbio.cnpem.br/en/divisoes-cientificas/nucleos-avancados-saude/descoberta-de-farmacos/)) of the Brazilian Biosciences National Laboratory (LNBio) from the Brazilian Center for Research in Energy and Materials (CNPEM) to empower natural products research with automation for biochemistry data processing and analysis. 


-------------------------------------------------------------

## Requirements

--------------------

The dependencies of all repositories are unified here in a single conda environment. Separated installation instructions may be found in the respective repository, if present.

- Ubuntu >= 20.04 (may also work with other Unix operating systems, but was not tested)
- Anaconda (https://www.anaconda.com/download/)
- CCP4 (with [Dimple](https://ccp4.github.io/dimple/))
- [Coot](https://www2.mrc-lmb.cam.ac.uk/personal/pemsley/coot/) - Crystallographic Object-Oriented Toolkit (comes with ccp4)
- GCC >= 7.4.0 and GCC <= 11
- Python >= 3.9 and packages
- Ubuntu packages:
  - build-essentials
  - libopenblas-dev
- For GPU use enabled: 
  - CUDA >= 10.1.243 and compatible with the CUDA version used for pytorch (e.g. if you use conda cudatoolkit=11.8, use CUDA=11.8 for MinkowskiEngine compilation)

---------------------------------

## Installation

----------------------

Tested in a Linux with Ubuntu 22.04.

Let's start with the installation of the Ubuntu packages and GCC 11:

```
sudo apt install build-essential libopenblas-dev g++-11
```

The required python and R packages will be installed with **anaconda + pip**. If you have any issues installing the packages, please report it on the github issue page.

The GPU compatibility is explained separated from the CPU only installation. In both cases the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) package, used for the deep learning model training and prediction, is installed at the end with pip.

### Anaconda + pip

Two pip requirements files are provided to help in the installation. One have GPU compatibility and the other is for CPU only.

First, we recommend setting the anaconda channel priority to flexible mode before creating the environment:
`conda config --set channel_priority true`

---------------------------------
#### Auto installation

Two scripts are provided for the CPU and the GPU installation. 
The Minkowski Engine python package, for DL training and validation, is installed by clonning its github source code in
the current folder and running the setup with corresponding parameters (preferred than the installation with pip).

For **CPU** only installation, run the script:

```
source install_conda_env_requirements_cpu.sh
```

For **GPU** enabled installation, there are additional requirement:
- CUDA >= 11.x and compatible with the CUDA version used for pytorch

The provided installation script uses a pytorch compatible with CUDA=11.8 and cuda-toolkit=11.8. 
For other CUDA versions please modify the corresponding script and requirements files with correct CUDA version. 
The pytorch CUDA version must match the cudtoolkit version.

For **GPU** enabled installation, run the script:
```
source install_conda_env_requirements_cuda11.8.sh
```

---------------------------------
#### Manual Installation

Create a conda environment with python 3.9 to encapsulate the installation and activate the environment:

```
conda create -n np3_lig python=3.9 -y
conda activate np3_lig
```

Then, install the openblas-devel package and the R base and packages:

```
conda install openblas-devel -c anaconda -y 
conda install r-base=4.4.0 r-readr r-dplyr -c conda-forge -y
```

And the anticlust R python package that is not present in conda.

```
Rscript -e 'install.packages("anticlust",repos = "http://cran.us.r-project.org")'
```

###### CPU only

Next, install the rest of the python packages requirements with pip:

```
pip install -r requirements_np3_ligand_cpu.txt
```

And finally install que Minkowski Engine package by cloning its github source code and running the setup for cpu only:

```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
export MAX_JOBS=2; # parallel compilation - prevent to much CPU assignment and process killed
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include:/usr/include/ --blas=openblas --cpu_only
```

###### GPU enabled

Additional requirement:
- CUDA >= 11.x and compatible with the CUDA version used for pytorch

The provided pip requirements files uses a pytorch compatible with CUDA=11.8 and cuda-toolkit=11.8. 
For other CUDA versions please modify the corresponding requirements .txt file and the following cudatoolkit version. 
The pytorch CUDA version must match the cudtoolkit version.

Check installed CUDA driver compatibility:
```
nvidia-smi
nvcc --version
```

Install the cuda=11.8 and cudatoolkit=11.8 with conda:

```
conda install nvidia/label/cuda-11.8.0::cuda
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y 
```

Next, install the rest of the python packages requirements with pip, here [pytorch](https://pytorch.org/get-started/previous-versions/) compatible with CUDA=11.8 is being used:

```
pip install -r requirements_np3_ligand_cuda11.8.txt --index-url https://download.pytorch.org/whl/cu118
```

Check the installed versions of pytorch, the corresponding CUDA used with it and GCC used, run:
```
python -c "import sys; import torch; print('Python version:', sys.version); print('Torch version:', torch.__version__); print('Torch CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); import MinkowskiEngine as ME; ME.print_diagnostics();" && gcc --version | head -n 1
```

And finally set the C++ compiler, set CUDA_HOME and install que Minkowski Engine by cloning its github source code and using the force_cuda parameter:

```
export CXX=g++-11;  # set this if you want to use a different C++ compiler
export CUDA_HOME=$(dirname $(dirname $(which nvcc))); # or select the correct cuda version on your system.
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH  
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
export MAX_JOBS=2; # parallel compilation - prevent to much CPU assignment and process killed
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas --force_cuda
```

---------------------------------
# How to use

-------------------------------

Before using the repositories functionalities the user must first activate the np3_lig environment.

```
conda activate np3_lig
```

More instructions are present in each repository documentation.

--------------------------------------
### Acknowledment

This research was funded by the [Serrapilheira](https://serrapilheira.org/en/) Institute, grant number Serra-1709-19681 (to Daniela B. B. Trivella).
It was part of the [Master's thesis](https://repositorio.unicamp.br/acervo/detalhe/1371294) of Cristina Freitas Bazzano, developed within an interdisciplinary project from the DDP-LNBio-CNPEM and the Institute of Computing from the University of Campinas (UNICAMP). 

--------------------------------------

### License
LigPCDS: Labeled Dataset of X-ray Protein Ligand Images in 3D Point Cloud Representations and Validated Deep Learning Models  © 2023 by Cristina Freitas Bazzano, Luiz F. G. Alves, Guilherme P. Telles, Daniela B. B. Trivella is licensed under CC BY 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/
