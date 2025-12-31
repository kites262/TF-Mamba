#!/bin/bash

conda install pytorch=2.8 -c pytorch -y
conda install mamba-ssm==2.2.6 -c conda-forge -y

pip install -e .
