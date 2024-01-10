# mpitree

![Build Status](https://github.com/ben-my-to/mpitree/workflows/Lint/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)

A Parallel Decision Tree Implementation using MPI *(Message Passing Interface)*.

## Requirements

- [mpi4py](https://pypi.org/project/mpi4py/) (>= 3.1.4)
- [numpy](https://pypi.org/project/pandas/) (>= 1.24.1)
- [matplotlib](https://pypi.org/project/matplotlib/) (>= 3.6.2)
- [scikit-learn](https://pypi.org/project/scikit-learn/) (>= 1.2.2)

## Installation

```bash
git clone https://github.com/ben-my-to/mpitree.git
cd mpitree && make build
```

## Example using `iris.py` with 2 processes

```bash
$ mpirun -n 2 python3 iris.py

┌── feature_0
│  ├── feature_1 [> 5.5]
│  │  └── class: 0 [> 3.6]
│  │  └── class: 2 [<= 3.6]
│  ├── feature_1 [> 5.5]
│  │  └── class: 0 [> 2.7]
│  │  └── class: 1 [<= 2.7]
Time: 3.20 ms
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Licence

[MIT](https://github.com/ben-my-to/mpitree/blob/main/LICENSE)
