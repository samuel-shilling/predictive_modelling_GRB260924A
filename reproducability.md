# Reproducibility Guide

This document provides instructions for downloading the repository, installing the required dependencies, and running the code.

## 1. Clone the repository

```bash
git clone https://github.com/samuel-shilling/predictive_modelling_GRB260924A.git
```


## 2. Navigate to the directory and install the dependencies and virtual environment 

Note, this virtual environment includes several big packages such as Redback which may need to be installed individually

Also, this environment includes lots of packages not needed for this project as it was carried over from another project

```bash
cd predictive_modelling_GRB260924A
conda env create -f environment.yml
conda activate venv_first
```

## 3. Run the code in terminal (can also be done in a jupyter notebook)

```bash
python src.py
```