# Quantum-to-classical correspondence in Krylov complexity
**Author**: Gastón F. Scialchi.
* Universidad de Buenos Aires, Facultad de Ciencias Exactas y Naturales, Departamento de Fı́sica.
* CONICET - Universidad de Buenos Aires, Instituto de Fı́sica de Buenos Aires (IFIBA).

This repository contains the numerical computations whose results appear in:

Gastón F. Scialchi, Augusto J. Roncaglia, and Diego A. Wisniacki. Quantum-to-classical correspondence in Krylov complexity. 2026. arXiv: 2603.11034 [quant-ph]. url: https://arxiv.org/abs/2603.11034.

This code is available under MIT license (see [LICENSE.md](./LICENSE.md)), you can use it freely.
If you do, please see the [Citation](#citation) section below.

## Installation
In order to run the scripts contained in this repository you will first need to create a python3 environment with the required dependencies.
You can do that by following the next steps:
* Clone the repository: ```git clone https://github.com/gscialchi/QtoC-Krylov.git ```
* Enter the directory: ```cd QtoC-Krylov```
* Create a virtual environment: ```python3 -m venv .venv```
* Activate the virtual environment: ```source .venv/bin/activate```
* Install the requirements: ```python3 -m pip install -r requirements.txt```

Now everything is ready to run the code.

## Reproducing figures
Make sure the virtual environment is active.
The following paths are relative to the repository's root directory.

In the scripts folders you'll find scripts that reproduce each figure in the manuscript.
You can run these scripts with

```python3 scripts/reproduce_fig_N.py```


You can run all scripts at once with

```python3 scripts/reproduce_all.py```

By default, a data and figures folder will be created in the repository's root directory.
This can be changed in the configuration file
(see [Configuration file](#configuration-file)).

Note that these calculations can take a while.
If you wish to get the figures quickly
you can download the precomputed data that appears in the paper,
see the [Data](#data) section below.

## Data
The code will automatically check whether
the computations have already been made and the outputs stored
in order to load them
(see [About the `store` and `doer` modules](#about-the-store-and-doer-modules)).

The full dataset that is produced by this code with the default configuration
(see [Configuration file](#configuration-file)),
and that appears in the figures present in the final version of the manuscript,
is available on Zenodo:

https://doi.org/10.5281/zenodo.19226991

Here are the steps to use it:
1. Download the dataset from the link above
2. Extract it in the repository's root directory: ```tar -xzvf data.tar.gz .```
3. As a result the whole directory structure should look like this:
```
QtoC-Krylov/
    configs/
    data/
        calculations/ # this is where 'doer' puts data
        operators/ # this is where 'store' puts data
    docs/
    scripts/
    src/
    tests
```
4. Alternatively, you can extract it elsewhere and edit the `DOER_DIR` and `STORE_DIR` variables in the configuration file.

You can now run the code as described above.

**Note:** the full dataset is 9 GB.

## Configuration file
The scripts can take an optional configuration file to modify the parameters for the calculations:

```python3 scripts/reproduce_fig_N.py --config=path/to/config.yml```

```python3 scripts/reproduce_all.py --config=path/to/config.yml```

If no configuration file is passed, configs/defaults.yml will be used, which contains the values used for the figures present in the final version of the manuscript.

## About the `store` and `doer` modules
Some computations can take a while, specially when array dimensions get somewhat large.
This is why I use two utilities that automatically check whether a given computation has already been made.
If it hasn't, then it computes and stores the data.
If it has, then it simply loads it.
The basic idea with both is to keep track of the methods and parameters involved and use that information to unically identify the data.

The `store` module is meant to be used on 'static' things that are used repeatedly, such as the operators that define the system's Hamiltonian, propagator, initial state, etc.
It provides a decorator that wraps the method used to compute said operators.

The `doer` module is meant to be used on 'calculation pipelines'.
It provides the `Doer` class which is to be used to define the 'pipeline elements'.
This is better explained by seeing it in action.

The results of the calculations do not depend on the use of these modules,
and they can be bypassed/deactivated completely by setting

```DISABLE_STORE: True```

and

```DISABLE_DOER: True```

in the configuration file,
which will result in the whole calculation being done from scratch.

## Citation
If you use the code from this repository in your research,
please cite:

https://doi.org/10.5281/zenodo.19226136

If you use the results of the paper, please cite:
```
@article{scialchi2026quantumtoclassicalcorrespondencekrylovcomplexity,
      title={Quantum-to-classical correspondence in Krylov complexity},
      author={Gastón F. Scialchi and Augusto J. Roncaglia and Diego A. Wisniacki},
      year={2026},
      eprint={2603.11034},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2603.11034},
}
```
