# mpitree

![Build Status](https://github.com/ben-my-to/mpitree/workflows/Unit%20Tests/badge.svg)
![Build Status](https://github.com/ben-my-to/mpitree/workflows/Lint/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)

A Parallel Decision Tree Implementation using MPI *(Message Passing Interface)*.

## Overview

![psplit](https://raw.githubusercontent.com/ben-my-to/mpitree/main/images/<REPLACE_WITH_IMAGE_NAME>.png)

For every interior decision tree node created, a variable number of processes collectively calculate the best feature to split *(i.e., the feature that maximizes the information gain)* Processes in a *communicator* are split approximately evenly across all levels of a split feature. Let $n$ be the total number of processes and $p$ be the number of levels, then each distribution, $m$ at some level $p$, contains at most $\lceil n/p \rceil$ processes and only one distribution contains $\lfloor n/p \rfloor$ processes where $n \nmid p$. Processes in a distribution independently participate among themselves at their respective levels. In summary, each process, $r$, is distributed in the cyclic distribution or round-robin fashion such that their $comm = r \mod p$ and $r' = \lfloor r/p \rfloor$.

Each routine waits for their respective processes from their original *communicator* to finish executing. The completion of a routine results in a sub-tree on a particular path from the root, and the local communicator is de-allocated. The algorithm terminates when all sub-trees are recursively gathered to the root process.

Note that all processes only perform a split during the *divide* phase in a given communicator at an interior node. Therefore, a leaf node may consist of more than one process, because the purity measurement at a node is independent of the number of processes.

In the above diagram, the root node consists of eight total processes, $p_0, p_1, ..., p_7$, with three distinct feature levels, $l_0, l_1, l_2$. Group 1 consists of processes and ranks, $(0,0), (1,1), (6,2), (7,3)$ respectively, Group 2 consists of processes and ranks, $(2,0), (3,1)$ respectively and Group 3 consists of processes and ranks, $(4,0), (5,1)$ respectively.

## Requirements

- [mpi4py](https://pypi.org/project/mpi4py/) (>= 3.1.4)
- [numpy](https://pypi.org/project/pandas/) (>= 1.24.1)
- [pandas](https://pypi.org/project/numpy/) (>= 1.5.2)
- [matplotlib](https://pypi.org/project/matplotlib/) (>= 3.6.2)
- [scikit-learn](https://pypi.org/project/scikit-learn/) (>= 1.2.2)

## Installation

```bash
git clone https://github.com/ben-my-to/mpitree.git
cd mpitree && make install
```

## Example using the *iris* dataset

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mpitree.decision_tree import ParallelDecisionTreeClassifier, WORLD_RANK

if __name__ == "__main__":
    iris = load_iris(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.20, random_state=42
    )

    # Concurrently train a decision tree classifier of `max_depth` 2 among all processes
    clf = ParallelDecisionTreeClassifier(criterion={"max_depth": 2})
    clf.fit(X_train, y_train)

    # Evaluate the performance (e.g., accuracy) of the decision tree classifier
    train_score, test_score = clf.score(X_train, y_train), clf.score(X_test, y_test)

    if not WORLD_RANK:
        print(clf)
        print(f"Train/Test Accuracy: ({train_score:.2%}, {test_score:.2%})")
```

### Executing `iris.py` with 5 processes

```bash
┌── petal length (cm)
│  └── 0 [< 2.45]
│  ├── petal length (cm) [>= 2.45]
│  │  └── 1 [< 4.75]
│  │  └── 2 [>= 4.75]
Train/Test Accuracy: (95.00%, 96.67%)
```

### Decision Boundaries varying values for the `max_depth` hyperparameter

Overfitting becomes apparent as the decision tree gets deeper because predictions are based on smaller and smaller cuboidal regions of the feature space. In a sense, the decision tree model is biasing towards *singleton* nodes; and, therefore, may cause mispredictions in the likelihood of noisy data.

Pre-and-post-pruning techniques are some solutions to reduce the likelihood of an overfitted decision tree. Pre-pruning techniques introduce early stopping criteria *(e.g., depth, number of samples)*. In both pruning techniques, one may resort to validation methodologies *(e.g., k-fold Cross-Validation)*.

The figure below depict various decision boundaries for different values of the `max_depth` hyperparameter. We used the *iris* dataset provided by *scikit-learn* as it provides a base analysis for our (parallel) decision tree implementation. The figure demonstrates how noisy instances may negatively impact the performance of the decision tree model.

![dt_noise](https://raw.githubusercontent.com/ben-my-to/mpitree/main/images/dt_noise.png)

## Unit Tests

```bash
pytest --doctest-modules
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Licence

[MIT](https://github.com/ben-my-to/mpitree/blob/main/LICENSE)
