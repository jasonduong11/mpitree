from sklearn.datasets import load_iris

from mpi4py import MPI
from mpitree.tree import ParallelDecisionTreeClassifier as pdt

iris = load_iris()
X, y = iris.data[:, :2], iris.target

start_time = MPI.Wtime()

# Concurrently train a decision tree classifier with max-depth 2 among all processes
clf = pdt(max_depth=2).fit(X, y)

if not pdt.WORLD_RANK:
    print(clf)
    print(f"Time: {MPI.Wtime() - start_time:.4f} secs")
