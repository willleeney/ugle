#!/bin/bash

echo "do you want to install a conda env? (y/other)"
read conda_option

if [ conda_option = y ]; then
    eval "$(conda shell.bash hook)"
    conda create -y -n ugle python=3.9.12
    conda activate ugle
fi

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip cache purge

python3 -m pip install torch==1.12.0

TORCH=$(python -c "import torch; print(torch.__version__)")
CUDA=$(python -c "import torch; print(torch.version.cuda)")

DGLCUDA=$(echo "${CUDA}")
CUDA=${CUDA//.}

if [ $CUDA = None ]; then
    DGLCUDA=''
    CUDA='cpu'
else
    DGLCUDA=$(echo "-cuda${DGLCUDA}")
    CUDA=$(echo "cu${CUDA}")
fi

TORCH=$(echo "${TORCH}" | cut -f1 -d"+")
echo "CUDA: ${CUDA}"
echo "TORCH: ${TORCH}"
echo "DGLCUDA: ${DGLCUDA}"

conda install -y -c dglteam dgl${DGLCUDA}

python3 -m pip install torch-scatter torch-sparse torchdiffeq torch-cluster torch-geometric -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html

python3 -m pip install -r requirements.txt
python3 -m pip install -e .
