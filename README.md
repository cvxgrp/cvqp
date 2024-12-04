# CVaR-Constrained Quadratic Programs (CVQP)

Implementation of the operator splitting method for solving large-scale
CVaR-constrained quadratic programs, as described in our paper "An Operator
Splitting Method for Large-Scale CVaR-Constrained Quadratic Programs".

## Installation and setup

This project uses [Poetry](https://python-poetry.org) for dependency management,
ensuring consistent environments and reliable package management across
different setups.

Install dependencies and create virtual environment:

```bash
poetry install
```

Activate the virtual environment:

```bash
poetry shell
```

## Usage 
To reproduce the results in the paper run the examples:
```bash
python examples.py
```
You can visualize the results in the notebook `examples.ipynb`.