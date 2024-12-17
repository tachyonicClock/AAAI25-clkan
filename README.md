# Kolmogorov-Arnold Networks Still Catastrophically Forget But Differently From MLP

This repository contains the code for the paper "Kolmogorov-Arnold Networks Still
Catastrophically Forget But Differently From MLP".

* Method PyTorch Lightning Modules are implemented in `src/clkan/regression/*`
* Pytorch Models are implemented in `src/clkan/models/*`
* The adaptive pruning procedure is implemented in `src/clkan/plugin/wisekan_plugin.py`
* Low level parameter isolation functions are implemented in `src/clkan/model/paraiso.py`

## Install

### Conda

Setup a conda virtual environment or python virtual environment with python3.12:

Using conda:

```bash
conda create -n clkan python=3.12
conda activate clkan
pip install -e .
```

### Venv

Using python and Ubuntu:

```bash
apt update && apt install python3.12-venv git
python3.12 -m venv clkan
source clkan/bin/activate
pip install -e .
```

### UV

Using [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12
uv sync
```

In later steps use `uv run` instead of `python -m`.

### Setup the datasets and environment variables

Setup a datasets directory using the `TORCH_DATA_DIR` environment variable:

```bash
export TORCH_DATA_DIR=/path/to/where/you/want/to/store/datasets
```

You can also set this in your `~/.bashrc` or `~/.bash_profile` to make it permanent.

Follow the instructions in the notebooks to download and prepare the datasets:

1. [feynman](dataset/feynman.ipynb)
2. [eurowind](dataset/eurowind.ipynb)
3. [riverradar](dataset/riverradar.ipynb)

## Software Stack

1. [Hydra](https://hydra.cc/) for hyper-parameter configuration. Will generate
   log files in the `multirun` directory and `outputs` directory.
2. [Optuna](https://optuna.org/) for hyper-parameter search. You can configure
   the backend for the search space using `OPTUNA_STORAGE` environment variable.
   It will default to `sqlite:///optuna.db`. Please note that when R2 is used in
   optuna for regression, it is actually the negative of the R2 score.
3. [Wandb](https://wandb.ai/) for logging experiments. You can configure the
   backend for the logging using `WANDB_LOG_ROOT` environment variable. It will
   default to `wandb`.
4. [PyTorch](https://pytorch.org/) for deep learning.
5. [PyTorch Lightning](https://www.pytorchlightning.ai/) for training loops.
6. `claiutil` a personal library for common utilities.
7. `efficientkan` a fork of
   [EfficientKAN](https://github.com/Blealtan/efficient-kan) adding
   normalization and linear input layers.

## Running Experiments

For each of the scenarios (`eurowind`, `feynman`, `riverradar`), you can run the
best hyper-parmeters configurations for any of the methods `ewc-kan`, `ewc-mlp`,
`joint-kan`, `joint-mlp`, `kan`, `mlp`, `packnet`, `si-kan`, `si-mlp`,
`wisekan`, `wisemlp`.

For example, to run the best hyper-parameters for the `riverradar` scenario
using the `wisekan` method, you can run:

```bash
python -m clkan +scenario/riverradar=wisekan # (for cpu add) +training.device=cpu
```

You may edit the configuration files in `src/clkan/conf/scenario/` to change the
hyper-parameters. Alternatively, you can override the hyper-parameters as command line
arguments. For example, to change the `lr` hyper-parameter for the `wisekan` method
in the `riverradar` scenario, you can run:

```bash
python -m clkan +scenario/riverradar=wisekan +training.lr=0.001
```

To get a full list of editable hyper-parameters you can add `--help` flag:

```bash
python -m clkan +scenario/riverradar=wisekan --help
```

## Running Hyper-Parameter Search

You can run a hyper-parameter search for any of the scenarios or methods. For example,
to run a hyper-parameter search for the `riverradar` scenario using the `wisekan` method,
you can run:

```bash
python -m clkan -m \
 +scenario/riverradar=wisekan \
 +sweep=wisekan \
 ++hydra.sweeper.n_jobs=2 \
 ++hydra.sweeper.n_trials=4
```

 > If you don't want to use WANDB for logging this might fail to run. To disable
 > `export WANDB_MODE="offline"` before running the command.

You may edit the search space in `src/clkan/conf/sweep/` to change the
hyper-parameters. Please refer to <https://hydra.cc/docs/plugins/optuna_sweeper/> for
more information on how to configure the search space.
