from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mpi4py import MPI
from mpitree.tree import ParallelDecisionTreeClassifier as pdt

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data[:, :2], iris.target, test_size=0.20, random_state=42
)

start_time = MPI.Wtime()

# Concurrently train a decision tree classifier of `max_depth` 3 among all processes
clf = pdt(max_depth=3).fit(X_train, y_train)

# Evaluate the performance (e.g., accuracy) of the decision tree classifier
train_score, test_score = clf.score(X_train, y_train), clf.score(X_test, y_test)

if not pdt.WORLD_RANK:
    print(clf)
    print(f"Train/Test Accuracy: ({train_score:.2%}, {test_score:.2%})")
    print(f"Time: {MPI.Wtime() - start_time:.4f} secs")
