#!/bin/bash
### Install python 3.9 using ANACONDA
###
export CXX=g++-9; # set this if you want to use a different C++ compiler
conda create -n np3_lig python=3.9 -y
conda activate np3_lig
# explicity set gcc equals 11 in conda env
conda install -c conda-forge gcc=9 gxx=9 -y
conda install openblas-devel -c anaconda -y
conda install r-base=4.4.0 r-readr r-dplyr -c conda-forge -y
# open3D dependency to draw geometrics
conda install -c conda-forge libstdcxx-ng=13.2 -y
## install R anticlust package - not present in conda                            
Rscript -e 'install.packages("anticlust",repos = "http://cran.us.r-project.org")'
## For CUDA capability also install cuda and cudatoolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda cuda-toolkit -y

# install pip requerements
pip install -r requirements_np3_ligand.txt --extra-index-url https://download.pytorch.org/whl/cu118

# install Minkowski Engine
export CXX=g++-9; # set this if you want to use a different C++ compiler
export CUDA_HOME=$(dirname $(dirname $(which nvcc))); # or select the correct cuda version on your system.
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH 
if [ ! -d 'lib/MinkowskiEngine' ]; then
  mkdir lib && cd lib
  git clone https://github.com/NVIDIA/MinkowskiEngine.git
  cd MinkowskiEngine
else
  cd lib/MinkowskiEngine
fi
export MAX_JOBS=2; # parallel compilation - prevent to much CPU assignment and process killed
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas --force_cuda
cd ../..
