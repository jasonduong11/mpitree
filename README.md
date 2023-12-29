# mpitree

![Build Status](https://github.com/ben-my-to/mpitree/workflows/Lint/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)

A Parallel Decision Tree Implementation using MPI *(Message Passing Interface)*.

## Overview

<table>
<tr>
    <td colspan="4" style=text-align:center;><b>Cyclic Distribution</b></td>
</tr>
<tr>
    <th style=background-color:#eee;><i>r</t></th>
    <td>0 1 2</td>
    <td>3 4 5</td>
    <td>6 7</td>
</tr>
<tr>
    <th style=background-color:#eee;><i>m</i></th>
    <td>0 1 2</td>
    <td>0 1 2</td>
    <td>0 1</td>
</tr>
<tr>
    <th style=background-color:#eee;><i>r<sub>m</sub></i></th>
    <td>0 0 0</td>
    <td>1 1 1</td>
    <td>2 2</td>
</tr>
</table>

The Parallel Decision Tree algorithm schedules processes in a cyclic distribution approximately evenly across levels of a split feature. Processes in each distribution independently participate among themselves at their respective levels and waits at their original *(parent)* communicator all other processes in that communicator. For every interior decision tree node created, a sub-communicator, $m$, is constructed, and each process per communicator concurrently participates in the calculation of the best feature to split *(i.e., the feature that maximizes the information gain)*. Let $n$ be the total number of processes and $p$ be the number of levels. Then, each distribution $m$ at some level $p$ contains at most $\lceil n/p \rceil$ processes, and only one distribution has $\lfloor n/p \rfloor$ processes where $n \nmid p$. Therefore, each process's rank $r$ is assigned to the sub-communicator $m = r \mod p$ and is assigned a unique rank in that group $r_m = \lfloor r/p \rfloor$.

A terminated routine call results in a sub-tree on a particular path from the root, and the *local* communicator is de-allocated. The algorithm terminates when all sub-trees are recursively gathered to the root process.

The table above shows a set of eight processes ranked $(0, 1, ..., 7)$, distributed across three feature levels. Group $0$ consists of a set of processes and ranks, $\{(0,0),(3,1),(6,2)\}$ respectively, Group $1$ consists of a set of processes and ranks, $\{(1,0),(4,1),(7,2)\}$ respectively and Group $2$ consists of a set of processes and ranks, $\{(2,0), (5,1)\}$ respectively.

## Requirements

- [mpi4py](https://pypi.org/project/mpi4py/) (>= 3.1.4)
- [numpy](https://pypi.org/project/pandas/) (>= 1.24.1)
- [matplotlib](https://pypi.org/project/matplotlib/) (>= 3.6.2)
- [scikit-learn](https://pypi.org/project/scikit-learn/) (>= 1.2.2)
- [graphviz](https://pypi.org/project/graphviz/) (>= 0.20.1)

## Installation

```bash
git clone https://github.com/ben-my-to/mpitree.git
cd mpitree && make install
```

## Example using `iris.py` with 2 processers

```bash
$ mpirun -n 2 python3 iris.py

┌── feature_0
│  ├── feature_1 [> 5.4]
│  │  ├── feature_0 [<= 3.9]
│  │  │  └── class: 2 [> 7.05]
│  │  │  └── class: 1 [<= 7.05]
│  │  └── class: 0 [> 3.9]
│  ├── feature_1 [<= 5.4]
│  │  ├── feature_0 [<= 2.8]
│  │  │  └── class: 1 [> 4.7]
│  │  │  └── class: 0 [<= 4.7]
│  │  └── class: 0 [> 2.8]
Train/Test Accuracy: (75.00%, 63.33%)
Time: 0.0338 secs
```

## Decision Boundaries varying values for the `max_depth` hyperparameter

Overfitting becomes apparent as the decision tree gets deeper because predictions are based on smaller and smaller cuboidal regions of the feature space. In a sense, the decision tree model is biasing towards *singleton* nodes; and, therefore, may cause mispredictions in the likelihood of noisy data.

Pre-and-post-pruning techniques are some solutions to reduce the likelihood of an overfitted decision tree. Pre-pruning techniques introduce early stopping criteria *(e.g., depth, number of samples)*. One may resort to validation methodologies *(e.g., k-fold Cross-Validation)* in both pruning techniques.

The figure below depicts various decision boundaries for different values of the `max_depth` hyperparameter. We used the *iris* dataset provided by *scikit-learn* as it gives a base analysis for our (parallel) decision tree implementation. The figure demonstrates how noisy instances may negatively impact the performance of the decision tree model.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Licence

[MIT](https://github.com/ben-my-to/mpitree/blob/main/LICENSE)
