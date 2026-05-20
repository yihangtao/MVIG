#!/usr/bin/env bash -l
set -e

env_name="advCP"

echo "########################## Conda Env ##########################"
if conda env list | grep -qE "^$env_name\s"; then
    echo "Conda environment '$env_name' exists."
else
    echo "Conda environment '$env_name' does not exist. creating ..."
    conda env create --name $env_name --file environment.yml
fi

echo "Installing pytorch, cuda, and related packages ..."
conda install -y --name $env_name python=3.7.11 pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# conda install -y --name $env_name python=3.7.11 pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# 设置 CUDA 相关环境变量
export CUDA_HOME=/data2/user2/yihang/cuda/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# 添加明确的 CUDA 架构标志
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# 验证版本
conda run -n $env_name python -c "import sys; print('Python version:', sys.version)"
conda run -n $env_name python -c "import torch; print('PyTorch version:', torch.__version__)"

# conda install -y --name $env_name pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# conda install -y --name $env_name pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=10.2 -c pytorch
pip install spconv-cu102

echo "########################## Dependency Setup ##########################"
echo "Setting up IoU cuda operator ..."
cd mvp/perception/cuda_op 
conda run --live-stream -n $env_name python setup.py install
cd -

echo "Setting up OpenCOOD ..."
cd third_party/OpenCOOD
rm -rf build/
rm -rf *.so
conda run --live-stream -n $env_name python opencood/utils/setup.py build_ext --inplace
conda run --live-stream -n $env_name python opencood/pcdet_utils/setup.py build_ext --inplace
cd -

echo "The environment is set and you should be able to run the evaluation. Commands"
echo "    conda activate advCP"
echo "    python scripts/evaluate.py"
