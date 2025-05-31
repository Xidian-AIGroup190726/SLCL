# TGRS-SLCL

doi: 10.1109/TGRS.2025.3574030

## PyTorch implementation of "Sample-Level Improved Cross-Source Contrastive Learning for PAN and MS Joint Classification".

![arch](https://github.com/user-attachments/assets/460c6f5a-851c-4a71-b3bc-bf8b61e31192)


## Installation
Clone the repository and run

    conda env create --name SLCL --file env.yml
    conda activate SLCL
    python pretrain.py

## Evaluation

We measure the quality of the learned representations by linear separability.

    python evaluation.py
