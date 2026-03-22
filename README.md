# Quantum-to-classical correspondence in Krylov complexity
Author: Gastón F. Scialchi.

This repository contains the numerical computations whose results appear in:

Gastón F. Scialchi, Augusto J. Roncaglia, and Diego A. Wisniacki. Quantum-to-classical correspondence in Krylov complexity. 2026. arXiv: 2603.11034 [quant-ph]. url: https://arxiv.org/abs/2603.11034.

## Installation
In order to run the scripts contained in this repository you will first need to create a python3 environment with the required dependencies.
You can do that by following the next steps:
* Clone the repository: ```git clone https://github.com/Sccial/quantum-to-classical-krylov-complexity.git ```
* Enter the directory: ```cd quantum-to-classical-krylov-complexity```
* Create a virtual environment: ```python3 -m venv .venv```
* Activate the virtual environment: ```source .venv/bin/activate```
* Install the requirements: ```python3 -m pip install -r requirements.txt```
Now everything is ready to run the code.

## Reproducing figures
Make sure the virtual environment is active.

In the scripts folders you'll find scripts that reproduce each figure in the manuscript.
You can run these scripts with ```python3 path/to/script/reproduce_fig_N.py```.
Each script can take an optional configuration file to modify the parameters for the calculations:
```python3 path/to/script/reproduce_fig_N.py --config=path/to/config.yml```.
If no configuration file is passed, configs/defaults.yml will be used, which contains the values used for the figures present in the final version of the manuscript.

By default, a data and figures folder will be created in the repository's root directory.
