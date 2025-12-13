#!/bin/bash

conda create -n pro@TF-Mamba python=3.11 -y
conda activate pro@TF-Mamba

conda install pytorch=2.8 -c pytorch -y
conda install mamba-ssm==2.2.6 -c conda-forge -y

pip install -e .
