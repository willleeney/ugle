#!/bin/bash

python3 -m pip install torch==1.12.0

TORCH=$(python -c "import torch; print(torch.__version__)")
CUDA=$(python -c "import torch; print(torch.version.cuda)")

CUDA=${CUDA//.}

if [ $CUDA = None ]; then
    CUDA='cpu'
else
    CUDA=$(echo "cu${CUDA}")
fi

TORCH=$(echo "${TORCH}" | cut -f1 -d"+")
echo "CUDA: ${CUDA}"
echo "TORCH: ${TORCH}"

python3 -m pip install torch-scatter torch-sparse torchdiffeq torch-cluster torch-geometric -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
