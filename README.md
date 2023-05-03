# NP³ Ligand

---------------------------------------

This repository stores the NP³ projects for ligand interpretation in X-ray protein crystallography. It contains three modules:

- **np3_Lig-PCDB**: the repository of the Lig-PCDB project with the code to create the ligands image databases and the stratified training dataset.
- **np3_DL_segmentation**: the repository with the training pipeline for semantic segmentation tasks used to validate Lig-PCDB and obtain the DL models.
- **np3_blob_label**: repository of the NP³ Blob Label application

The NP³ is a project from the Drug Discovery Platform ([DDP](https://lnbio.cnpem.br/innovation-core/technological-platforms/drug-discovery/)) of the Brazilian Biosciences National Laboratory (LNBio) from the Brazilian Center for Research in Energy and Materials (CNPEM) to empower natural products research with automations for biochemistry data processing and analysis. 

-------------------------------------------------------------
-------------------------------------------------------------

## Requirements

--------------------

The dependencies of all repositories are unified here in a single environment. Separated installation instructions may be found in the respective repository, if present.

- Ubuntu >= 14.04 (may also work with other Unix operating systems, but was not tested)
- Anaconda (https://www.anaconda.com/download/)
- CCP4 (with [Dimple](https://ccp4.github.io/dimple/))
- [Coot](https://www2.mrc-lmb.cam.ac.uk/personal/pemsley/coot/) - Crystallographic Object-Oriented Toolkit
- GCC >= 7.4.0
- Python >= 3.8 and packages
- Ubuntu packages:
  - build-essentials
  - libopenblas-dev
- For GPU compatibility: 
  - CUDA >= 10.1.243 and compatible with the CUDA version used for pytorch (e.g. if you use conda cudatoolkit=11.1, use CUDA=11.1 for MinkowskiEngine compilation)

## Installation

----------------------

Let's start with the installation of the Ubuntu packages and GCC 7:

```
sudo apt install build-essential libopenblas-dev g++-7
```

The required python and R packages will be installed with **anaconda + pip**. If you have any issues installing the packages, please report it on the github issue page.

The GPU compatibility is explained separated from the CPU only installation. In both cases the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) package, used for the deep learning model training and prediction, is installed at the end with pip.

### Anaconda + pip

Two pip requirements files are provided to help in the installation. One have GPU compatibility and the other is for CPU only.

First, we recommend setting the anaconda channel priority to flexible mode before creating the environment:
`conda config --set channel_priority true`

Create a conda environment with python 3.8 to encapsulate the installation and activate the environment:

```
conda create -n np3_lig python=3.8
conda activate np3_lig
```

Then, install the openblas package and the R base and packages:

```
conda install openblas-devel -c anaconda
conda install r-base=3.6.3 r-readr r-dplyr -c conda-forge
```

And the anticlust R python package that is not present in conda.

```
Rscript -e 'install.packages("anticlust",repos = "http://cran.us.r-project.org")'
```

#### CPU only

Next, install the rest of the python packages requirements with pip:

```
pip install -r requirements_np3_ligand_cpu.txt
```

And finally install que Minkowski Engine package with pip:

```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas=openblas" --install-option="--cpu_only"
```

#### GPU compatibility

Additional requirement:
- CUDA >= 11.1 and compatible with the CUDA version used for pytorch

The provided pip requirements files uses a pytorch compatible with CUDA>=11.1 and cuda-toolkit=11.1. For other CUDA versions please modify the corresponding requirements .txt file and the following cudatoolkit version. The pytorch CUDA version must match the cudtoolkit version.

Install the cudatoolkit=11.1 with conda:

```
conda install cudatoolkit=11.1 -c pytorch -c conda-forge 
```

Next, install the rest of the python packages requirements with pip, here [pytorch](https://pytorch.org/get-started/previous-versions/) compatible with CUDA=11.1 is being used:

```
pip install -r requirements_np3_ligand_cuda11.1.txt -f https://download.pytorch.org/whl/torch_stable.html
```

And finally set the C++ compiler, set CUDA_HOME and install que Minkowski Engine with pip and the force_cuda parameter:

```
export CXX=g++-7;  # set this if you want to use a different C++ compiler
export CUDA_HOME=$(dirname $(dirname $(which nvcc))); # or select the correct cuda version on your system.
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas=openblas" --install-option="--force_cuda"
```

---------------------------------
----------------------------------

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
It was part of the Master's thesis of Cristina Freitas Bazzano, developed within an interdisciplinary project from the DDP-LNBio-CNPEM and the Institute of Computing from the University of Campinas (UNICAMP). 
