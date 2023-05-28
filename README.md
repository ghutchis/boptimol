# BOptiMol

This repo contains code for optimizing molecules using Bayesian optimization and quantum chemistry computations.

The aim is to perform accelerated local search using an active
learning approach. That is, the optimization will guess final
bond lengths, angles, and dihedrals based on previous
optimizations.

At the moment, the project is experimental.

## Installation

Build the environment using anaconda:

```bash
mamba env create --file environment.yml --force
```

## Use

`run.py` provides a simple interface to the code. To optimize cysteine with default arguments. For now, the code expects molecules in XYZ format.

```bash
python run.py test/molecules/peroxide.xyz
```

