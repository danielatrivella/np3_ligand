### Install python 3.9 using ANACONDA
###
export CXX=g++-11; 
conda create -n np3_lig python=3.9 -y
conda activate np3_lig
# explicity set gcc equals 11 in conda env
conda install -c conda-forge gcc=11.2.0 gxx=11.2.0 -y
conda install openblas-devel -c anaconda -y
conda install r-base=4.4.0 r-readr r-dplyr -c conda-forge -y
# open3D dependency to draw geometrics
conda install -c conda-forge libstdcxx-ng=13.2 -y
## install R anticlust package - not present in conda                            
Rscript -e 'install.packages("anticlust",repos = "http://cran.us.r-project.org")' 
# install pip requerements
pip install -r requirements_np3_ligand_cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

# install Minkowski Engine
export CXX=g++-11; # set this if you want to use a different C++ compiler
if [ ! -d 'MinkowskiEngine' ]; then 
  git clone https://github.com/NVIDIA/MinkowskiEngine.git
fi
cd MinkowskiEngine
export MAX_JOBS=2; # parallel compilation - prevent to much CPU assignment and process killed
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include:/usr/include/ --blas=openblas --cpu_only
