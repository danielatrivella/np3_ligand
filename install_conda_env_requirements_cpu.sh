### Install python 3.9 using ANACONDA
###
conda create -n np3_lig python=3.9 -y
conda activate np3_lig
conda install openblas-devel -c anaconda -y
conda install r-base=4.4.0 r-readr r-dplyr -c conda-forge -y
## install R anticlust package - not present in conda                            
Rscript -e 'install.packages("anticlust",repos = "http://cran.us.r-project.org")' 
# install pip requerements
pip install -r requirements_np3_ligand_cpu.txt --index-url https://download.pytorch.org/whl/cu118

# install Minkowski Engine
export CXX=g++-9; # set this if you want to use a different C++ compiler
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
export MAX_JOBS=2; # parallel compilation - prevent to much CPU assignment and process killed
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include:/usr/include/ --blas=openblas --cpu_only
